"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import dynamic from 'next/dynamic';
import { useMarketData } from "@/lib/hooks/useMarketData";
import { useNewsData } from "@/lib/hooks/useNewsData";
import { useAppState } from "@/lib/context";
import { Activity, DollarSign, Brain } from "lucide-react";
import { VirtualWalletCard } from "@/components/VirtualWalletCard";
import { AgentStatusBanner } from "@/components/AgentStatusBanner";
import { NewsScannerWidget } from "@/components/NewsScannerWidget";
import { NewsInsightsWidget } from "@/components/NewsInsightsWidget";

// Use dynamic import for TradingView widget because it relies on window/document
const TradingChart = dynamic(() => import('@/components/TradingChart'), { ssr: false });

const availableCoins = ["BTCUSDT", "ETHUSDT", "AAVEUSDT", "SOLUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"];

export default function Dashboard() {
  const [symbol, setSymbol] = React.useState("ALL");
  // Main data hook for the active chart symbol (defaults to BTCUSDT if ALL is selected)
  const activeChartSymbol = symbol === "ALL" ? "BTCUSDT" : symbol;
  const { klines, orderbook, tradeHistory } = useMarketData(activeChartSymbol);
  const { rawNews, predictions } = useNewsData();
  const { status, signals, currency, exchangeRates } = useAppState();

  const rate = exchangeRates[currency] || 1;
  const currencySymbol = currency === 'GBP' ? '£' : currency === 'EUR' ? '€' : currency === 'JPY' ? '¥' : '$';

  // Fetch prices for all coins if "ALL" is selected
  const [allPrices, setAllPrices] = React.useState<Record<string, { current: number, diff: number, pct: number }>>({});
  
  React.useEffect(() => {
    if (symbol !== "ALL") return;
    
    // Initial fetch for 24h stats to calculate diffs
    const fetchStats = async () => {
      try {
        const res = await fetch('https://api.binance.com/api/v3/ticker/24hr?' + new URLSearchParams({
          symbols: JSON.stringify(availableCoins)
        }));
        const data = await res.json();
        const initialPrices: any = {};
        for (const d of data) {
          const current = parseFloat(d.lastPrice);
          const diff = parseFloat(d.priceChange);
          const pct = parseFloat(d.priceChangePercent);
          initialPrices[d.symbol] = { current, diff, pct };
        }
        setAllPrices(initialPrices);
      } catch (e) { console.error(e); }
    };
    fetchStats();

    // Subscribe to live combined streams
    const streams = availableCoins.map(c => `${c.toLowerCase()}@ticker`).join('/');
    const ws = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${streams}`);
    
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload?.data?.s) {
          const d = payload.data;
          setAllPrices(prev => ({
            ...prev,
            [d.s]: {
              current: parseFloat(d.c),
              diff: parseFloat(d.p),
              pct: parseFloat(d.P)
            }
          }));
        }
      } catch (e) {
        console.error(e);
      }
    };

    return () => ws.close();
  }, [symbol]);

  // Single active coin info
  const lastKline = klines.length > 0 ? klines[klines.length - 1] : null;
  const prevKline = klines.length > 2 ? klines[klines.length - 2] : null;
  const currentPrice = lastKline ? (lastKline.close ?? lastKline[4] ?? 0) : 0;
  const prevPrice = prevKline ? (prevKline.close ?? prevKline[4] ?? 0) : 0;
  
  const parsedCurrent = parseFloat(currentPrice?.toString() || "0") * rate;
  const parsedPrev = parseFloat(prevPrice?.toString() || "0") * rate;
  const priceDiff = parsedCurrent - parsedPrev;
  const priceDiffPct = parsedPrev > 0 ? (priceDiff / parsedPrev) * 100 : 0;

  const latestSignal = signals?.[0];

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto pt-4 font-sans fadeIn">
      <div className="flex flex-wrap gap-2 mb-4">
        {["ALL", ...availableCoins].map(coin => (
          <button
            key={coin}
            onClick={() => setSymbol(coin)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${symbol === coin ? 'bg-zinc-100 text-zinc-900' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'}`}
          >
            {coin.replace('USDT', '')}
          </button>
        ))}
      </div>

      <div className={`grid gap-6 ${symbol === "ALL" ? "md:grid-cols-2" : "md:grid-cols-3"}`}>

        {symbol === "ALL" ? (
          <Card className="col-span-1 md:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle>All Markets</CardTitle>
              <DollarSign size={15} className="text-zinc-400" />
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
                {availableCoins.map(coin => {
                  const data = allPrices[coin] || { current: 0, diff: 0, pct: 0 };
                  return (
                    <div key={coin} className="p-3 bg-[#121214] border border-zinc-800/50 rounded-xl">
                      <div className="text-xs text-zinc-400 font-medium mb-1">{coin.replace('USDT', '')}</div>
                      <div className="text-lg tracking-tight font-semibold text-[#D1D4DC] font-mono">
                        {currencySymbol}{(data.current * rate).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className={`text-[11px] font-mono mt-1 ${data.diff >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                        {data.diff >= 0 ? '+' : ''}{(data.diff * rate).toFixed(2)} ({data.diff >= 0 ? '+' : ''}{data.pct.toFixed(2)}%)
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle>Current Price</CardTitle>
              <DollarSign size={15} className="text-zinc-400" />
            </CardHeader>
            <CardContent>
              <div className="text-xl tracking-tight font-semibold text-[#D1D4DC] font-mono">
                {currencySymbol}{parsedCurrent.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <p className={`text-[12px] font-mono mt-1 font-medium ${priceDiff >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                {priceDiff >= 0 ? '+' : ''}{priceDiff.toFixed(2)} ({priceDiffPct.toFixed(2)}%)
              </p>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle>AI Conviction</CardTitle>
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
          </CardHeader>
          <CardContent>
            <div className="text-lg tracking-tight font-semibold text-[#D1D4DC] flex items-center">
              Neuromorphic Engine
            </div>
            <p className="text-[12px] tracking-wide text-zinc-400 mt-1 capitalize">Mode: Standing By</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="col-span-1 md:col-span-2 grid gap-6 md:grid-cols-2">
           <VirtualWalletCard />
        </div>
        <div className="col-span-1 md:col-span-2">
           <AgentStatusBanner />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-7 lg:grid-cols-8">
        <Card className="col-span-4 lg:col-span-6 min-h-[500px] flex flex-col p-0 overflow-hidden">
          <CardHeader className="border-b border-zinc-800/50">
            <CardTitle>Market Trajectory</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 p-0 m-0">
            <TradingChart symbol={activeChartSymbol} currencyRate={rate} currencyPrefix={currencySymbol} />
          </CardContent>
        </Card>

        <Card className="col-span-3 lg:col-span-2 h-full">
          <CardHeader className="border-b border-zinc-800/50 pb-4">
            <CardTitle>Orderbook Alpha</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-[11px] text-zinc-400 space-y-4 pt-4">
            <div>
              <div className="flex justify-between mb-2 text-zinc-400">
                <span>Price ({currency})</span>
                <span>Size ({activeChartSymbol.replace('USDT', '')})</span>
              </div>
              <div className="space-y-1.5 mt-2">
                {orderbook?.asks?.slice(0, 5).reverse().map((ask: number[], i: number) => (
                  <div key={i} className="flex justify-between text-rose-500/90">
                    <span>{(parseFloat(ask[0].toString()) * rate).toFixed(2)}</span>
                    <span className="text-[#D1D4DC]">{parseFloat(ask[1].toString()).toFixed(4)}</span>
                  </div>
                ))}
                <div className="my-3 border-y border-zinc-800/50 py-1.5 text-center text-[#D1D4DC] tracking-widest text-[10px]">
                  SPREAD
                </div>
                {orderbook?.bids?.slice(0, 5).map((bid: number[], i: number) => (
                  <div key={i} className="flex justify-between text-emerald-500/90">
                    <span>{(parseFloat(bid[0].toString()) * rate).toFixed(2)}</span>
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
                    <span>{currencySymbol}{(parseFloat(trade.price) * rate).toFixed(1)}</span>
                    <span className="text-[#D1D4DC]">{new Date(trade.time).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })}</span>
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
