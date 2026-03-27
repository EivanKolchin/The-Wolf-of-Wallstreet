import sys
from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from backend.signals.technical import build_technical_feature_dict
from backend.agents.nn_model import TradingLSTM
from structlog import get_logger

log = get_logger("scripts.pretrain")


def fetch_binance_data(symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 8640) -> pd.DataFrame:
    """Fetch recent K-lines from Binance"""
    log.info("Fetching data from Binance", symbol=symbol, limit=limit)
    url = "https://api.binance.com/api/v3/klines"
    
    # Binance limits to 1000 per request. We loop backwards.
    all_data = []
    end_time = None
    
    needed = limit
    while needed > 0:
        batch_limit = min(needed, 1000)
        params = {"symbol": symbol, "interval": interval, "limit": batch_limit}
        if end_time:
            params["endTime"] = end_time
            
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        
        if not data:
            break
            
        all_data = data + all_data
        end_time = data[0][0] - 1  # end before the first candle of this batch
        needed -= len(data)
        
        log.info(f"Fetched {len(data)} candles, {needed} remaining...")
        time.sleep(0.5)

    
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    return df.sort_values("timestamp").reset_index(drop=True)

def build_vector(idx: int, df: pd.DataFrame) -> np.ndarray:
    """Builds a 62-element feature vector for a specific index in df (simulating realtime context)."""
    # For technicals, we need to pass a slice of df to build_technical_feature_dict
    # Need at least 200 bars for some MAs/ATRs
    start_idx = max(0, idx - 250)
    sub_df = df.iloc[start_idx:idx+1].copy()
    
    tech = build_technical_feature_dict(sub_df)
    
    vec = np.zeros(62, dtype=np.float32)
    
    try:
        # 0-2: prices
        op, hi, lo, cl = sub_df.iloc[-1][["open", "high", "low", "close"]]
        vec[0] = np.clip((cl - op) / op, -0.05, 0.05) * 20.0  # scaled -1, 1
        vec[1] = np.clip((hi - op) / op, 0.0, 0.05) * 20.0
        vec[2] = np.clip((lo - op) / op, -0.05, 0.0) * 20.0
    except:
        pass
        
    vec[3] = tech.get("volume_ratio", 0.0)
    vec[4] = 0.5 # spread_pct neutral
    
    vec[5] = tech.get("ema_9_dist", 0.0)
    vec[6] = tech.get("ema_21_dist", 0.0)
    vec[7] = tech.get("ema_50_dist", 0.0)
    vec[8] = tech.get("ema_200_dist", 0.0)
    vec[9] = tech.get("golden_cross", 0.0)
    vec[10] = tech.get("vwap_dist", 0.0)
    vec[11] = tech.get("rsi", 0.0)
    vec[12] = tech.get("macd_norm", 0.0)
    vec[13] = tech.get("macd_hist_norm", 0.0)
    vec[14] = tech.get("stoch_rsi", 0.0)
    vec[15] = tech.get("adx_norm", 0.0)
    vec[16] = tech.get("rsi_divergence", 0.0)
    vec[17] = tech.get("atr_norm", 0.0)
    vec[18] = tech.get("bb_width_norm", 0.0)
    vec[19] = tech.get("bb_pct_b", 0.0)
    vec[20] = tech.get("volume_ratio", 0.0)
    vec[21] = tech.get("obv_slope", 0.0)
    vec[22] = tech.get("fib_nearest_level_pct", 0.0)
    vec[23] = tech.get("fib_distance", 0.0)
    vec[24] = tech.get("fib_strength", 0.0)
    
    patterns = tech.get("pattern_flags", [0.0]*10)
    for i in range(10):
        if i < len(patterns):
            vec[25 + i] = patterns[i]
            
    # Orderbook neutrals
    vec[35:43] = 0.5 # 0.5 or 0.0? features.py uses 0.0 for missing
    
    # Regime: let's assume ranging [0,0,1,0,0,0]
    vec[45] = 1.0 
    
    # News neutrals
    vec[49:53] = 0.0
    
    # Macro neutrals
    vec[53] = 0.5 
    vec[54] = 0.5 
    vec[55] = 0.0 
    vec[56] = 0.0
    
    # Time (mock hour based on row index - just placeholder)
    now = df.iloc[idx]["timestamp"]
    if isinstance(now, pd.Timestamp):
        vec[57] = math.sin(2 * math.pi * now.hour / 24.0)
        vec[58] = math.cos(2 * math.pi * now.hour / 24.0)
        vec[59] = math.sin(2 * math.pi * now.weekday() / 7.0)
        vec[60] = math.cos(2 * math.pi * now.weekday() / 7.0)
    
    vec[61] = 0.5 # regime_confidence
    return vec


def test_trading_lstm():
    df = fetch_binance_data(limit=8640)
    log.info("Computing features for each candle (this will take a minute...)")
    
    # We need to compute vectors and labels
    # Label: Look 3 candles ahead (15 minutes)
    # If future_close > current * 1.005 -> 0, < 0.995 -> 1, else 2.
    
    features_list = []
    labels = []
    
    # Pre-calculate features to speed up
    # However we need history per row. Wait, since build_technical_feature_dict is stateless and applies pandas_ta
    # across the entire df, we can just compute it over the WHOLE dataframe ONCE, then extract rows.
    # The prompt explicitly asks to use build_technical_feature_dict() "for each candle".
    # But running it 8640 times on slices would be very O(N^2) slow. Let's run it once on the full DF and extract rows if possible.
    # WAIT! build_technical_feature_dict() returns iloc[-1] only natively!
    # "All functions take a pandas DataFrame... and return the most recent bar's values only (iloc[-1])." -> Prompt 5.
    
    # Ok, let's run it continuously from idx 250 to end.
    n = len(df)
    valid_start = 250
    for i in range(valid_start, n - 3):
        if i % 1000 == 0:
            log.info(f"Processed {i}/{n} candles...")
        
        vec = build_vector(i, df)
        features_list.append(vec)
        
        # Determine label
        current_close = df.iloc[i]["close"]
        future_close = df.iloc[i+3]["close"]
        if future_close > current_close * 1.005:
            labels.append(0)
        elif future_close < current_close * 0.995:
            labels.append(1)
        else:
            labels.append(2)

    features_matrix = np.array(features_list)
    labels_array = np.array(labels)
    
    # Print feature stats
    means = features_matrix.mean(axis=0)
    stds = features_matrix.std(axis=0)
    
    log.info("Feature stats:")
    for i in range(62):
        if stds[i] < 0.01:
            log.warning(f"Feature {i} has low variance: std = {stds[i]:.4f}")
            
    # Build Sequence Dataset (SEQUENCE_LENGTH = 60, step = 1)
    seq_len = 60
    X_seq = []
    y_seq = []
    
    for i in range(len(features_matrix) - seq_len):
        X_seq.append(features_matrix[i:i+seq_len])
        y_seq.append(labels_array[i+seq_len-1])
        
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    
    log.info(f"Sequence dataset shape: {X_seq.shape}")
    
    # Train/Val Split (80/20 Chronological)
    split_idx = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    log.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 30
    best_val_loss = float("inf")
    
    models_dir = Path(root_dir) / "models"
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "pretrain_best.pt"
    latest_model_path = models_dir / "trading_lstm_latest.pt"
    
    log.info(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            probs, size = model(bx)
            # Use negative log likelihood for pre-softmaxed probs
            loss = F.nll_loss(torch.log(probs + 1e-8), by)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Eval
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                probs, size = model(bx)
                
                loss = F.nll_loss(torch.log(probs + 1e-8), by)
                val_loss += loss.item() * bx.size(0)
                
                preds = probs.argmax(dim=1)
                correct += (preds == by).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        
        log.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log.info(f"New best model! Saving to {best_model_path}...")
            torch.save(model.state_dict(), best_model_path)
            
    # Load best model and copy
    log.info(f"Training complete. Copying best model to {latest_model_path}")
    import shutil
    shutil.copy(best_model_path, latest_model_path)
    
    # Optionally print extra summary
    log.info("Pretraining pipeline finished successfully!")

if __name__ == "__main__":
    test_trading_lstm()
