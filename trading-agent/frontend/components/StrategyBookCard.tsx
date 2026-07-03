"use client";

import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { RefreshCcw } from "lucide-react";
import { cn } from "@/lib/utils";

interface Allocation {
  symbol: string;
  weight: number;
  value: number;
  side: "long" | "short";
}
interface HistoryPoint {
  ts: number;
  value: number;
}
interface StrategyPortfolio {
  active: boolean;
  total_value: number;
  initial_value: number;
  cash: number;
  gross_exposure: number;
  net_exposure: number;
  total_pnl: number;
  total_pnl_pct: number;
  allocations: Allocation[];
  history: HistoryPoint[];
  paper?: boolean;
  updated_at?: number;
}

const fmtUsd = (n: number) =>
  `$${(n ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const cleanSym = (s: string) => s.replace("USDT", "").replace("-USD", "");

export function StrategyBookCard() {
  const [data, setData] = useState<StrategyPortfolio | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchBook() {
      try {
        const res = await fetch("http://localhost:8000/api/strategy/portfolio");
        if (!res.ok) throw new Error("failed");
        setData(await res.json());
      } catch {
        /* keep last data on a transient error */
      } finally {
        setLoading(false);
      }
    }
    fetchBook();
    const id = setInterval(fetchBook, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading && !data) {
    return (
      <Card className="col-span-2">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle>Strategy Book</CardTitle>
          <RefreshCcw className="w-4 h-4 animate-spin text-zinc-500" />
        </CardHeader>
        <CardContent className="h-40 flex items-center justify-center">
          <p className="text-zinc-500">Loading portfolio…</p>
        </CardContent>
      </Card>
    );
  }

  const active = data?.active && (data?.history?.length ?? 0) > 0;
  const pnl = data?.total_pnl ?? 0;
  const isUp = pnl >= 0;

  const chartData =
    data?.history?.map((h) => ({
      t: new Date(h.ts * 1000).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
      }),
      value: h.value,
    })) ?? [];

  const stroke = isUp ? "#10b981" : "#f43f5e";

  return (
    <Card className="col-span-2">
      <CardHeader className="pb-4 border-b border-zinc-800/50 flex flex-row items-center justify-between">
        <CardTitle>
          Strategy Book
          <span className="ml-2 text-[10px] uppercase tracking-widest rounded-sm bg-zinc-800 px-1.5 py-0.5 text-zinc-400 align-middle">
            {data?.paper === false ? "LIVE" : "PAPER"}
          </span>
        </CardTitle>
        <span className="text-[10px] uppercase tracking-widest text-zinc-600 font-mono">
          Managed-beta + TS-momentum
        </span>
      </CardHeader>

      <CardContent className="pt-6">
        {!active ? (
          <div className="h-32 flex flex-col items-center justify-center gap-1 text-center">
            <p className="text-zinc-400 text-sm">The strategy book has not rebalanced yet.</p>
            <p className="text-zinc-600 text-xs font-mono">
              It appears within one rebalance of the agent starting (STRATEGY_AGENT_ENABLED=true).
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Total value hero */}
              <div className="col-span-1 border-r-0 lg:border-r border-zinc-800/50 pr-0 lg:pr-6 flex flex-col justify-center">
                <p className="text-[12px] tracking-wide text-zinc-400 mb-1">Total Portfolio Value</p>
                <h2 className="text-3xl font-semibold tracking-tight text-[#D1D4DC] font-mono">
                  {fmtUsd(data!.total_value)}
                </h2>
                <div
                  className={cn(
                    "mt-2 text-[12px] font-mono font-medium",
                    isUp ? "text-emerald-500" : "text-rose-500"
                  )}
                >
                  {isUp ? "+" : ""}
                  {fmtUsd(pnl)} ({isUp ? "+" : ""}
                  {(data!.total_pnl_pct ?? 0).toFixed(2)}%) since start
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] font-mono text-zinc-500">
                  <span>Gross exp: {((data!.gross_exposure ?? 0) * 100).toFixed(0)}%</span>
                  <span>Net exp: {((data!.net_exposure ?? 0) * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Net-worth line chart */}
              <div className="col-span-1 lg:col-span-2 h-44">
                <p className="text-[11px] tracking-wider text-zinc-400 uppercase mb-2">
                  Net Worth Over Time
                </p>
                <ResponsiveContainer width="100%" height="90%">
                  <AreaChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="nw" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={stroke} stopOpacity={0.25} />
                        <stop offset="100%" stopColor={stroke} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f1f22" vertical={false} />
                    <XAxis
                      dataKey="t"
                      tick={{ fontSize: 9, fill: "#71717a" }}
                      minTickGap={40}
                      axisLine={{ stroke: "#27272a" }}
                      tickLine={false}
                    />
                    <YAxis
                      domain={["auto", "auto"]}
                      tick={{ fontSize: 9, fill: "#71717a" }}
                      width={54}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0e0e10",
                        border: "1px solid #27272a",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                      labelStyle={{ color: "#a1a1aa" }}
                      formatter={(v) => [fmtUsd(Number(v)), "Net worth"]}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke={stroke}
                      strokeWidth={2}
                      fill="url(#nw)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Allocations incl. cash */}
            <div className="mt-6 pt-4 border-t border-zinc-800/50">
              <p className="text-[11px] font-semibold tracking-wider text-zinc-400 uppercase mb-3">
                Allocations
              </p>
              <div className="space-y-2">
                {/* Cash row */}
                <div className="flex items-center justify-between p-2 rounded-lg bg-zinc-900/60 border border-zinc-800">
                  <span className="text-[12px] font-bold text-zinc-300">Cash / Uninvested</span>
                  <div className="flex items-center gap-4">
                    <span className="text-[11px] font-mono text-zinc-500">
                      {(((data!.cash ?? 0) / (data!.total_value || 1)) * 100).toFixed(1)}%
                    </span>
                    <span className="text-[12px] font-mono font-semibold text-zinc-300 w-28 text-right">
                      {fmtUsd(data!.cash)}
                    </span>
                  </div>
                </div>
                {data!.allocations.length === 0 ? (
                  <p className="text-xs font-mono text-zinc-600">No open allocations</p>
                ) : (
                  data!.allocations.map((a) => (
                    <div
                      key={a.symbol}
                      className="flex items-center justify-between p-2 rounded-lg bg-zinc-900 border border-zinc-800"
                    >
                      <span className="text-[12px] font-bold text-zinc-200">
                        {cleanSym(a.symbol)}
                        <span
                          className={cn(
                            "text-[10px] ml-2 uppercase rounded-sm px-1.5 py-0.5",
                            a.side === "long"
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-rose-500/20 text-rose-400"
                          )}
                        >
                          {a.side}
                        </span>
                      </span>
                      <div className="flex items-center gap-4">
                        <span className="text-[11px] font-mono text-zinc-500">
                          {(Math.abs(a.weight) * 100).toFixed(1)}%
                        </span>
                        <span
                          className={cn(
                            "text-[12px] font-mono font-semibold w-28 text-right",
                            a.value >= 0 ? "text-[#D1D4DC]" : "text-rose-400"
                          )}
                        >
                          {fmtUsd(a.value)}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="mt-5 pt-3 border-t border-zinc-800/50 text-[10px] text-zinc-500 flex justify-between uppercase tracking-widest font-mono">
              <span>Base: {fmtUsd(data!.initial_value)}</span>
              <span>
                {data?.updated_at
                  ? `Updated ${new Date(data.updated_at * 1000).toLocaleTimeString()}`
                  : ""}
              </span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
