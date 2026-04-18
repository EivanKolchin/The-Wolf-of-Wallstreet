"use client";

import React, { useEffect, useState } from 'react';

interface OpenTrade {
  symbol: string;
  entry_price: number;
  current_price: number;
  highest_price_seen: number;
  trailing_stop_limit: number;
  unrealised_pnl: number;
}

export default function TrailingStopTable() {
  const [trades, setTrades] = useState<OpenTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const res = await fetch('/api/open-trades');
        if (!res.ok) throw new Error('Failed to fetch open trades');
        const data = await res.json();
        setTrades(data);
        setError(null);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 3000); // poll every 3 seconds

    return () => clearInterval(interval);
  }, []);

  const getRowStatus = (trade: OpenTrade) => {
    // Assuming long positions for stop logic
    if (trade.current_price <= trade.trailing_stop_limit) {
      return 'breached';
    }
    
    const diffPct = (trade.current_price - trade.trailing_stop_limit) / trade.trailing_stop_limit;
    if (diffPct <= 0.005) {
      return 'warning';
    }
    
    return 'safe';
  };

  const getRowClasses = (status: string) => {
    switch (status) {
      case 'breached':
        return 'bg-rose-950/40 hover:bg-rose-900/40 border-l-2 border-rose-500';
      case 'warning':
        return 'bg-yellow-950/30 hover:bg-yellow-900/40 border-l-2 border-yellow-500';
      default:
        return 'bg-slate-900/20 hover:bg-slate-800/40 border-l-2 border-transparent';
    }
  };

  const formatCurrency = (val: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 6 }).format(val);

  if (loading && trades.length === 0) {
    return (
      <div className="w-full p-6 flex justify-center items-center rounded-xl border border-slate-800 bg-slate-950/50">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="w-full rounded-xl border border-slate-800 bg-slate-950 shadow-lg overflow-hidden flex flex-col">
      <div className="p-4 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
          Trailing Stop Monitor
        </h2>
        {error ? (
          <span className="text-xs text-rose-400 font-medium">Connection error</span>
        ) : (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            Live Monitoring
          </div>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-slate-400 uppercase bg-slate-900/80 border-b border-slate-800">
            <tr>
              <th className="px-4 py-3 font-medium">Symbol</th>
              <th className="px-4 py-3 font-medium">Entry Price</th>
              <th className="px-4 py-3 font-medium">Current Price</th>
              <th className="px-4 py-3 font-medium">Peak</th>
              <th className="px-4 py-3 font-medium">Trailing Stop</th>
              <th className="px-4 py-3 font-medium text-right">Unrealised PnL</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50 font-mono text-[13px]">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-slate-500 font-sans">
                  No active trades monitored.
                </td>
              </tr>
            ) : (
              trades.map((trade, idx) => {
                const status = getRowStatus(trade);
                const pnlIsPositive = trade.unrealised_pnl >= 0;

                return (
                  <tr key={`${trade.symbol}-${idx}`} className={`transition-colors ${getRowClasses(status)}`}>
                    <td className="px-4 py-3 font-bold text-slate-200 flex items-center gap-2">
                      <span className="relative flex h-1.5 w-1.5">
                        {status !== 'breached' && (
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                        )}
                        <span className={`relative inline-flex rounded-full h-1.5 w-1.5 ${status === 'breached' ? 'bg-rose-500' : 'bg-sky-500'}`}></span>
                      </span>
                      {trade.symbol}
                    </td>
                    <td className="px-4 py-3 text-slate-400">{formatCurrency(trade.entry_price)}</td>
                    <td className={`px-4 py-3 font-semibold ${status === 'warning' ? 'text-yellow-400' : status === 'breached' ? 'text-rose-400' : 'text-slate-200'}`}>
                      {formatCurrency(trade.current_price)}
                    </td>
                    <td className="px-4 py-3 text-sky-400">{formatCurrency(trade.highest_price_seen)}</td>
                    <td className="px-4 py-3 text-slate-300 border-l border-slate-800/50">
                      {formatCurrency(trade.trailing_stop_limit)}
                    </td>
                    <td className={`px-4 py-3 text-right font-bold ${pnlIsPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {pnlIsPositive ? '+' : ''}{formatCurrency(trade.unrealised_pnl)}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
