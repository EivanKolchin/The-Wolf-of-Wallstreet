"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import dynamic from 'next/dynamic';
import { useMarketData } from "@/lib/hooks/useMarketData";
import { useNewsData } from "@/lib/hooks/useNewsData";
import { useAppState } from "@/lib/context";
import { Activity, DollarSign, Brain } from "lucide-react";
import { NewsScannerWidget } from "@/components/NewsScannerWidget";
import { NewsInsightsWidget } from "@/components/NewsInsightsWidget";

// Use dynamic import for TradingView widget because it relies on window/document
const TradingChart = dynamic(() => import('@/components/TradingChart'), { ssr: false });

export default function Dashboard() {
  const { klines, orderbook, tradeHistory } = useMarketData("btcusdt");
  const { rawNews, predictions } = useNewsData();
  const { status, signals } = useAppState();

  const currentPrice = klines.length > 0 ? klines[klines.length - 1].close : 0;
  const prevPrice = klines.length > 2 ? klines[klines.length - 2].close : 0;
  const priceDiff = currentPrice - prevPrice;
  const priceDiffPct = prevPrice > 0 ? (priceDiff / prevPrice) * 100 : 0;

  const latestSignal = signals?.[0];

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto pt-4 font-sans fadeIn">
      <div className="grid gap-6 md:grid-cols-3">

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle>Current Price</CardTitle>
            <DollarSign size={15} className="text-zinc-400" />
          </CardHeader>
          <CardContent>
            <div className="text-xl tracking-tight font-semibold text-[#D1D4DC] font-mono">
              ${parseFloat(currentPrice.toString()).toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
            <p className={`text-[12px] font-mono mt-1 font-medium ${priceDiff >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
              {priceDiff >= 0 ? '+' : ''}{priceDiff.toFixed(2)} ({priceDiffPct.toFixed(2)}%)
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle>AI Conviction</CardTitle>
            <Activity size={15} className="text-zinc-400" />
          </CardHeader>
          <CardContent>
            <div className="text-lg tracking-tight font-semibold text-[#D1D4DC] flex items-center capitalize">
              {latestSignal?.direction || "Neutral"}
            </div>
            <p className={`text-[12px] tracking-wide mt-1 font-medium ${latestSignal?.direction === 'long' ? 'text-emerald-500' : latestSignal?.direction === 'short' ? 'text-rose-500' : 'text-zinc-400'}`}>
              Confidence: {Math.round((latestSignal?.metadata?.confidence || 0) * 100)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle>LLM Analysis</CardTitle>
            <Brain size={15} className="text-zinc-400" />
          </CardHeader>
          <CardContent>
            <div className="text-lg tracking-tight font-semibold text-[#D1D4DC] flex items-center">
              Neuromorphic Engine
            </div>
            <p className="text-[12px] tracking-wide text-zinc-400 mt-1 capitalize">Mode: Standing By</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-7 lg:grid-cols-8">
        <Card className="col-span-4 lg:col-span-6 min-h-[500px] flex flex-col p-0 overflow-hidden">
          <CardHeader className="border-b border-zinc-800/50">
            <CardTitle>Market Trajectory</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 p-0 m-0">
            <TradingChart symbol="BTCUSDT" />
          </CardContent>
        </Card>

        <Card className="col-span-3 lg:col-span-2 h-full">
          <CardHeader className="border-b border-zinc-800/50 pb-4">
            <CardTitle>Orderbook Alpha</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-[11px] text-zinc-400 space-y-4 pt-4">
            <div>
              <div className="flex justify-between mb-2 text-zinc-400">
                <span>Price (USD)</span>
                <span>Size (BTC)</span>
              </div>
              <div className="space-y-1.5 mt-2">
                {orderbook?.asks?.slice(0, 5).reverse().map((ask: number[], i: number) => (
                  <div key={i} className="flex justify-between text-rose-500/90">
                    <span>{parseFloat(ask[0].toString()).toFixed(2)}</span>
                    <span className="text-[#D1D4DC]">{parseFloat(ask[1].toString()).toFixed(4)}</span>
                  </div>
                ))}
                <div className="my-3 border-y border-zinc-800/50 py-1.5 text-center text-[#D1D4DC] tracking-widest text-[10px]">
                  SPREAD
                </div>
                {orderbook?.bids?.slice(0, 5).map((bid: number[], i: number) => (
                  <div key={i} className="flex justify-between text-emerald-500/90">
                    <span>{parseFloat(bid[0].toString()).toFixed(2)}</span>
                    <span className="text-[#D1D4DC]">{parseFloat(bid[1].toString()).toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="pt-4 mt-4 border-t border-zinc-800/50">
               <div className="flex justify-between mb-3 text-zinc-400">
                <span>Recent Prints</span>
                <span>Time</span>
              </div>
              <div className="space-y-2 opacity-90">
                {tradeHistory?.slice(0, 8).map((trade: any, i: number) => (
                  <div key={i} className={`flex justify-between ${trade.is_buyer_maker ? 'text-rose-500' : 'text-emerald-500'}`}>
                    <span>${parseFloat(trade.price).toFixed(1)}</span>
                    <span className="text-[#D1D4DC]">{new Date(trade.time).toLocaleTimeString([], { hour12: false, second: '2-digit' })}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <div className="col-span-1 border-t-0">
          <NewsScannerWidget rawNews={rawNews} />
        </div>
        <div className="col-span-1 lg:col-span-2 border-t-0">
          <NewsInsightsWidget predictions={predictions} />
        </div>
      </div>
    </div>
  );
}
