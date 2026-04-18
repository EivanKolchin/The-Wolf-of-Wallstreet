"use client";

import React, { useEffect, useState } from 'react';

interface ReputationData {
  agent_id: string;
  win_rate: number;
  total_pnl: number;
  total_trades: number;
  last_updated: string;
}

const CircularProgress = ({ value }: { value: number }) => {
  const radius = 24;
  const circumference = 2 * Math.PI * radius;
  // Cap at 100 for offset calculation
  const safeValue = Math.min(Math.max(value, 0), 100);
  const strokeDashoffset = circumference - (safeValue / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg className="w-14 h-14 transform -rotate-90">
        <circle
          className="text-slate-800"
          strokeWidth="4"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx="28"
          cy="28"
        />
        <circle
          className={safeValue >= 50 ? "text-emerald-500" : "text-rose-500"}
          strokeWidth="4"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx="28"
          cy="28"
        />
      </svg>
      <div className="absolute text-xs font-bold text-slate-200">
        {value.toFixed(1)}%
      </div>
    </div>
  );
};

export default function ReputationCard() {
  const [data, setData] = useState<ReputationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Using relative URL to hit the Next.js proxy/backend
        const res = await fetch('/api/reputation');
        if (!res.ok) throw new Error('Failed to fetch reputation');
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // refresh every minute
    return () => clearInterval(interval);
  }, []);

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.max(0, Math.floor(diffMs / 60000));
    if (diffMins === 0) return 'Just now';
    return `${diffMins} min${diffMins !== 1 ? 's' : ''} ago`;
  };

  if (loading && !data) {
    return (
      <div className="w-full max-w-xs p-4 rounded-xl border border-slate-800 bg-slate-950/50 flex items-center justify-center min-h-[140px]">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="w-full max-w-xs p-4 rounded-xl border border-red-900/50 bg-slate-950/50 min-h-[140px]">
        <p className="text-xs text-rose-500 text-center">Data unavailable</p>
      </div>
    );
  }

  const isPositive = data.total_pnl >= 0;

  return (
    <div className="w-full max-w-xs p-4 rounded-xl border border-slate-800 bg-slate-950 flex flex-col gap-3 shadow-lg">
      
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-1.5">
          <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" /></svg>
          Agent Reputation
        </h3>
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">{data.agent_id}</span>
      </div>

      {/* Main Stats Row */}
      <div className="flex items-center justify-between my-1">
        <div className="flex flex-col">
          <span className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold mb-1">Total PnL</span>
          <span className={`text-xl font-bold font-mono ${isPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
            {isPositive ? '+' : ''}{data.total_pnl.toLocaleString('en-US', { style: 'currency', currency: 'USD' })}
          </span>
        </div>
        
        <div className="flex flex-col items-center">
          <span className="text-[11px] text-slate-500 uppercase tracking-wider font-semibold mb-1">Win Rate</span>
          <CircularProgress value={data.win_rate} />
        </div>
      </div>

      {/* Footer / Meta Row */}
      <div className="flex items-center justify-between mt-1 pt-3 border-t border-slate-800/80">
        <div className="text-[11px] font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/50">
          <span className="text-slate-400 mr-1">Trades:</span>
          {data.total_trades}
        </div>
        <div className="text-[10px] text-slate-500 italic flex items-center gap-1">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
          {formatTimeAgo(data.last_updated)}
        </div>
      </div>

    </div>
  );
}
