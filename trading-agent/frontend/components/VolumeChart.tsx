"use client";

import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface VolumeData {
  date: string;
  defi_volume: number;
  cex_volume: number;
}

export default function VolumeChart() {
  const [data, setData] = useState<VolumeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchVolume = async () => {
      try {
        const res = await fetch('/api/trade-volume-split');
        if (!res.ok) throw new Error('Failed to fetch volume data');
        const json = await res.json();
        setData(json);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchVolume();
  }, []);

  if (loading) {
    return (
      <div className="w-full h-80 flex justify-center items-center rounded-xl border border-slate-800 bg-slate-950/50">
        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  if (error || data.length === 0) {
    return (
      <div className="w-full h-80 flex flex-col justify-center items-center rounded-xl border border-red-900/50 bg-slate-950/50">
        <p className="text-xs text-rose-500 font-medium">Data unavailable</p>
      </div>
    );
  }

  // Calculations for summary row
  const totalDefi = data.reduce((acc, curr) => acc + curr.defi_volume, 0);
  const totalCex = data.reduce((acc, curr) => acc + curr.cex_volume, 0);
  const grandTotal = totalDefi + totalCex;
  const defiPercent = grandTotal > 0 ? (totalDefi / grandTotal) * 100 : 0;
  const cexPercent = grandTotal > 0 ? (totalCex / grandTotal) * 100 : 0;

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg shadow-xl shrink-0">
          <p className="text-slate-300 text-xs mb-2 font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center justify-between gap-4 text-xs font-mono">
              <span className="flex items-center gap-1.5" style={{ color: entry.color }}>
                <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: entry.color }} />
                <span className="font-sans font-semibold">{entry.name === 'defi_volume' ? 'DeFi' : 'CEX'}</span>
              </span>
              <span className="font-bold text-slate-200">
                {formatCurrency(entry.value)}
              </span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full p-5 rounded-xl border border-slate-800 bg-slate-950 shadow-lg flex flex-col">
      <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <svg className="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
        30-Day Volume Split
      </h2>

      <div className="flex-1 min-h-[220px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            {/* Minimalist setup: no grid, no axis lines, subtle ticks */}
            <XAxis 
              dataKey="date" 
              axisLine={false} 
              tickLine={false} 
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={(val) => {
                const d = new Date(val);
                return `${d.getMonth() + 1}/${d.getDate()}`;
              }}
              minTickGap={20}
            />
            <YAxis 
              axisLine={false} 
              tickLine={false} 
              tick={{ fill: '#64748b', fontSize: 10 }}
              tickFormatter={(val) => `$${(val / 1000).toFixed(0)}k`}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#1e293b', opacity: 0.4 }} />
            
            {/* Stacked bars */}
            <Bar dataKey="defi_volume" stackId="a" fill="#6366f1" radius={[0, 0, 2, 2]} />
            <Bar dataKey="cex_volume" stackId="a" fill="#64748b" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Row */}
      <div className="mt-5 pt-4 border-t border-slate-800/80 grid grid-cols-3 gap-4">
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Total Volume</span>
          <span className="text-sm font-bold font-mono text-slate-200">{formatCurrency(grandTotal)}</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-indigo-500/80 uppercase tracking-wider font-semibold flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div> DeFi
          </span>
          <span className="text-sm font-bold font-mono text-indigo-400">{defiPercent.toFixed(1)}%</span>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-slate-500"></div> CEX
          </span>
          <span className="text-sm font-bold font-mono text-slate-400">{cexPercent.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}
