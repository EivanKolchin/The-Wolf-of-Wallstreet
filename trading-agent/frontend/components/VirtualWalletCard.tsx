"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { RefreshCcw } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";

interface PortfolioResponse {
  initial_usdc: number;
  available_cash: number;
  locked_cash: number;
  realized_pnl: number;
  unrealized_pnl: number;
  total_value: number;
  live_positions: Array<{
    symbol: string;
    unrealized: number;
    size_usd: number;
  }>;
}

export function VirtualWalletCard() {
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    async function fetchPortfolio() {
      try {
        const res = await fetch("http://localhost:8000/api/portfolio");
        if (!res.ok) throw new Error("Failed to fetch");
        const json = await res.json();
        setData(json);
        setLastUpdate(new Date());
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    fetchPortfolio();
    const interval = setInterval(fetchPortfolio, 3000); // 3 second live polling
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle>Virtual Ledger</CardTitle>
          <RefreshCcw className="w-4 h-4 animate-spin text-zinc-500" />
        </CardHeader>
        <CardContent className="h-40 flex items-center justify-center">
            <p className="text-zinc-500">Retrieving Ledgers...</p>
        </CardContent>
      </Card>
    );
  }

  if (!data) return null;

  const realized = data.realized_pnl ?? 0;
  const unrealized = data.unrealized_pnl ?? 0;
  const totalPnL = realized + unrealized;
  const isPositive = totalPnL >= 0;

  return (
    <Card className="col-span-2">
      <CardHeader className="flex flex-row items-center justify-between pb-4 border-b border-zinc-800/50">
        <CardTitle>Virtual Ledger</CardTitle>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            <span className="text-[10px] font-medium text-emerald-500 uppercase tracking-widest">Live Sync</span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Total Value Hero */}
          <div className="col-span-1 border-r-0 lg:border-r border-zinc-800/50 pr-0 lg:pr-6 flex flex-col justify-center">
            <p className="text-[12px] tracking-wide text-zinc-400 mb-1">Total Portfolio Value</p>
            <h2 className="text-3xl font-semibold tracking-tight text-[#D1D4DC] font-mono flex items-center gap-1">
              ${(data.total_value ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </h2>
            <div className={cn("mt-2 flex items-center gap-1 text-[12px] font-mono font-medium", isPositive ? "text-emerald-500" : "text-rose-500")}>
                {isPositive ? "+" : ""}
                ${totalPnL.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} All Time
            </div>
          </div>

          {/* Breakdown Grid */}
          <div className="col-span-1 lg:col-span-2 grid grid-cols-2 gap-4">
            <div className="p-3 bg-[#0A0A0A] rounded-lg border border-zinc-800/50">
              <p className="text-[11px] tracking-wider text-zinc-400 uppercase mb-1">Available Cash</p>
              <p className="text-lg font-mono font-semibold text-[#D1D4DC]">${(data.available_cash ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            </div>
            <div className="p-3 bg-[#0A0A0A] rounded-lg border border-zinc-800/50">
              <p className="text-[11px] tracking-wider text-zinc-400 uppercase mb-1">Locked Margin</p>
              <p className="text-lg font-mono font-semibold text-[#D1D4DC]">${(data.locked_cash ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            </div>
            <div className="p-3 bg-[#0A0A0A] rounded-lg border border-zinc-800/50">
               <p className="text-[11px] tracking-wider text-zinc-400 uppercase mb-1">Realized PnL</p>
               <p className={cn("text-lg font-mono font-semibold", (data.realized_pnl ?? 0) >= 0 ? "text-emerald-500" : "text-rose-500")}>
                  {(data.realized_pnl ?? 0) >= 0 ? "+" : ""}${(data.realized_pnl ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
               </p>
            </div>
            <div className="p-3 bg-[#0A0A0A] rounded-lg border border-zinc-800/50">
               <p className="text-[11px] tracking-wider text-zinc-400 uppercase mb-1">Unrealized PnL</p>
               <p className={cn("text-lg font-mono font-semibold", (data.unrealized_pnl ?? 0) >= 0 ? "text-emerald-500" : "text-rose-500")}>
                  {(data.unrealized_pnl ?? 0) >= 0 ? "+" : ""}${(data.unrealized_pnl ?? 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
               </p>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-zinc-800/50 text-[10px] text-zinc-500 flex justify-between uppercase tracking-widest font-mono">
          <span>Base Capital: ${(data.initial_usdc ?? 0).toLocaleString()} USDC</span>
          <span>Sync: {formatDistanceToNow(lastUpdate, { addSuffix: true })}</span>
        </div>
      </CardContent>
    </Card>
  );
}