import os
import shutil
import random
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TradeExperience:
    features_sequence: np.ndarray  # (sequence_length, 62)
    direction_taken: int           # 0=long, 1=short, 2=hold
    actual_pnl_pct: float
    
    @property
    def reward(self) -> float:
        return math.tanh(self.actual_pnl_pct * 10)

class TradingLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=62, 
            hidden_size=128, 
            num_layers=2, 
            dropout=0.2, 
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4, 
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.fc_direction = nn.Linear(128, 3)
        self.fc_size = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, sequence_length, 62)
        lstm_out, _ = self.lstm(x) # (batch, seq, 128)
        
        # Self-attention over LSTM outputs
        # MultiheadAttention with batch_first=True expects query, key, value as (batch, seq, feature)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last timestep
        last_step = attn_out[:, -1, :] # (batch, 128)
        last_step = self.dropout(last_step)
        
        # Direction output mapping to [long_prob, short_prob, hold_prob]
        direction_logits = self.fc_direction(last_step)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Size output
        size_logits = self.fc_size(last_step)
        size = torch.sigmoid(size_logits)
        
        return direction_probs, size

class ReplayBuffer:
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: TradeExperience) -> None:
        self.buffer.append(experience)

    def sample(self, n: int) -> list[TradeExperience]:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

class PersistentTradingModel:
    MODEL_PATH = Path("models/trading_lstm_latest.pt")
    CHECKPOINT_DIR = Path("models/checkpoints/")
    SEQUENCE_LENGTH = 60

    def __init__(self):
        self.model = TradingLSTM()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.replay_buffer = ReplayBuffer(max_size=10_000)
        
        self.trade_count = 0
        self.cumulative_pnl = 0.0
        
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        self._load_or_initialise()

    def _load_or_initialise(self):
        if self.MODEL_PATH.exists():
            checkpoint = torch.load(self.MODEL_PATH, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.trade_count = checkpoint.get('trade_count', 0)
            self.cumulative_pnl = checkpoint.get('cumulative_pnl', 0.0)
            
            logger.info("model_loaded", 
                        trade_count=self.trade_count, 
                        cumulative_pnl=self.cumulative_pnl,
                        path=str(self.MODEL_PATH))
        else:
            logger.info("first_run_initialising_model")
            self._pretrain_on_synthetic_data()
            self.safe_checkpoint(label="initial")

    def safe_checkpoint(self, label: str = ""):
        tmp_path = self.MODEL_PATH.with_suffix(".tmp")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'trade_count': self.trade_count,
            'cumulative_pnl': self.cumulative_pnl
        }
        
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(self.MODEL_PATH)
        
        ckpt_filename = f"ckpt_{self.trade_count}{'_' + label if label else ''}.pt"
        ckpt_path = self.CHECKPOINT_DIR / ckpt_filename
        shutil.copy(self.MODEL_PATH, ckpt_path)
        
        # Keep only the last 10 checkpoints
        checkpoints = sorted(self.CHECKPOINT_DIR.glob("ckpt_*.pt"), key=os.path.getmtime)
        if len(checkpoints) > 10:
            for old_ckpt in checkpoints[:-10]:
                old_ckpt.unlink()
                
        # DB update happens via application logic tracking agent events rather than synchronous write here
        logger.info("model_checkpoint_saved", path=str(self.MODEL_PATH), trade_count=self.trade_count)

    def infer(self, feature_sequence: np.ndarray) -> tuple[str, float, dict]:
        # Input shape expected: (SEQUENCE_LENGTH, 62)
        seq_tensor = torch.tensor(feature_sequence, dtype=torch.float32).unsqueeze(0) # (1, seq_len, 62)
        
        self.model.eval()
        with torch.no_grad():
            direction_probs, size = self.model(seq_tensor)
            
            # Extract to scalars
            probs_arr = direction_probs[0].numpy()
            raw_size = size[0].item()
            
            probs_dict = {
                "long": float(probs_arr[0]),
                "short": float(probs_arr[1]),
                "hold": float(probs_arr[2])
            }
            
            decision_idx = int(np.argmax(probs_arr))
            decision_map = {0: "long", 1: "short", 2: "hold"}
            decision = decision_map[decision_idx]
            
            position_size_pct = float(np.clip(raw_size, 0.02, 0.20))
            
            return decision, position_size_pct, probs_dict

    def online_update(self, experience: TradeExperience) -> None:
        self.replay_buffer.add(experience)
        self.trade_count += 1
        self.cumulative_pnl += experience.actual_pnl_pct
        
        if self.trade_count % 10 == 0 and len(self.replay_buffer) >= 32:
            batch = self.replay_buffer.sample(32)
            
            seqs = np.stack([ex.features_sequence for ex in batch])
            sequences_t = torch.tensor(seqs, dtype=torch.float32)
            
            targets = []
            for ex in batch:
                # actual_better_direction logic
                if ex.actual_pnl_pct > 0:
                    targets.append(ex.direction_taken)
                elif ex.actual_pnl_pct < 0:
                    # opposite direction
                    if ex.direction_taken == 0:
                        targets.append(1)
                    elif ex.direction_taken == 1:
                        targets.append(0)
                    else:
                        targets.append(2)
                else:
                    targets.append(2)
                    
            targets_t = torch.tensor(targets, dtype=torch.long)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            pred_probs, _ = self.model(sequences_t)
            # Use negative log likelihood since forward applies softmax natively
            loss = F.nll_loss(torch.log(pred_probs + 1e-8), targets_t)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
        if self.trade_count % 50 == 0:
            self.safe_checkpoint()

    def check_and_rollback(self, recent_pnl_pct: float, threshold: float = -0.05) -> bool:
        if recent_pnl_pct < threshold:
            checkpoints = sorted(self.CHECKPOINT_DIR.glob("ckpt_*.pt"), key=os.path.getmtime)
            if len(checkpoints) >= 2:
                target_ckpt = checkpoints[-2] # second most recent
                logger.warning("pnl_threshold_breached_triggering_rollback", target_checkpoint=str(target_ckpt))
                
                checkpoint = torch.load(target_ckpt, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.trade_count = checkpoint.get('trade_count', self.trade_count)
                self.cumulative_pnl = checkpoint.get('cumulative_pnl', self.cumulative_pnl)
                
                # Replace latest with fallback state safely
                self.safe_checkpoint(label="rollback")
                return True
        return False

    def _pretrain_on_synthetic_data(self):
        # Generate 1000 synthetic feature sequences
        # Batch size 32 -> ~31 batches per epoch
        self.model.train()
        n_samples = 1000
        n_batches = n_samples // 32
        
        for epoch in range(5):
            for _ in range(n_batches):
                # (batch, seq, 62) - random close to 0.5
                X_batch_np = np.random.normal(0.5, 0.1, (32, self.SEQUENCE_LENGTH, 62)).astype(np.float32)
                X_batch = torch.tensor(X_batch_np)
                
                # Label: hold (2)
                y_batch = torch.full((32,), 2, dtype=torch.long)
                
                self.optimizer.zero_grad()
                pred_probs, _ = self.model(X_batch)
                loss = F.nll_loss(torch.log(pred_probs + 1e-8), y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
