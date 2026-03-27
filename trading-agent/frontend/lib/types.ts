export interface Trade {
    id: string;
    asset: string;
    direction: "long" | "short";
    size_usd: number;
    entry_price: number;
    current_price?: number;
    unrealised_pnl?: number;
    stop_loss: number;
    take_profit: number;
    kite_tx_hash?: string;
    prediction_score?: number;
}

export interface NewsImpact {
    severity: "NEUTRAL" | "SIGNIFICANT" | "SEVERE";
    asset: string;
    direction: "up" | "down" | "neutral";
    magnitude_pct_low: number;
    magnitude_pct_high: number;
    confidence: number;
    t_min_minutes: number;
    t_max_minutes: number;
    rationale: string;
    source_domain: string;
    trust_score: number;
}

export interface PortfolioStatus {
    total_value_usd: number;
    available_cash: number;
    unrealised_pnl: number;
    daily_pnl: number;
    drawdown_pct: number;
    peak_value: number;
    is_halted: boolean;
}

