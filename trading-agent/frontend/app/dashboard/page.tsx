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
// Restricted stock universe (mirrors backend/core/universe.py STOCK_UNDERLYINGS)
const availableStocks = ["SNDK", "AMD", "MU", "AXTI", "BE"];

export default function Dashboard() {
  const [symbol, setSymbol] = React.useState("ALL");
  // Asset-class mode: crypto / stocks / both
  const [assetClass, setAssetClass] = React.useState<"crypto" | "stocks" | "both">("crypto");
  // Phase 12: per-symbol attention state (HIGH/LOW) + manual overrides (from /api/attention)
  const [attentionState, setAttentionState] = React.useState<Record<string, "high" | "low">>({});
  const [attentionOverrides, setAttentionOverrides] = React.useState<Record<string, "high" | "low">>({});
  const displayedSymbols = assetClass === "crypto" ? availableCoins
    : assetClass === "stocks" ? availableStocks
    : [...availableCoins, ...availableStocks];
  const isStock = (s: string) => availableStocks.includes(s);
  // Reset to ALL when switching asset class — but only if the current symbol
  // isn't already valid for the new class (e.g. the user picked AMD from the
  // chart's dropdown which also flips the class; we don't want to clobber that).
  React.useEffect(() => {
    setSymbol(prev => {
      if (prev === "ALL") return prev;
      const validHere = (assetClass === "stocks" && availableStocks.includes(prev))
                     || (assetClass === "crypto" && availableCoins.includes(prev))
                     || (assetClass === "both");
      return validHere ? prev : "ALL";
    });
  }, [assetClass]);
  // Main data hook for the active chart symbol (defaults to the first symbol of the class)
  const activeChartSymbol = symbol === "ALL" ? (assetClass === "stocks" ? availableStocks[0] : "BTCUSDT") : symbol;

  // Phase 12: poll /api/attention every 5s so the badges reflect live agent state.
  React.useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/attention");
        const data = await res.json();
        if (cancelled) return;
        setAttentionState(data.state || {});
        setAttentionOverrides(data.overrides || {});
      } catch { /* keep last good values */ }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Cycle override: auto -> HIGH -> LOW -> auto (null = clear)
  const cycleAttention = async (coin: string) => {
    const cur = attentionOverrides[coin];
    const next: "high" | "low" | null = cur === undefined
      ? "high"
      : cur === "high" ? "low" : null;
    try {
      await fetch("http://127.0.0.1:8000/api/attention", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: coin, attention: next }),
      });
      // optimistic update — server is authoritative; next poll will reconcile
      setAttentionOverrides((prev) => {
        const cp = { ...prev };
        if (next === null) delete cp[coin]; else cp[coin] = next;
        return cp;
      });
    } catch { /* swallow; next poll reconciles */ }
  };
  const { klines, orderbook, tradeHistory } = useMarketData(activeChartSymbol);
  const { rawNews, predictions } = useNewsData();
  const { status, signals, currency, exchangeRates } = useAppState();

  const rate = exchangeRates[currency] || 1;
  const currencySymbol = currency === 'GBP' ? '£' : currency === 'EUR' ? '€' : currency === 'JPY' ? '¥' : '$';

  // Fetch prices for all coins if "ALL" is selected
  const [allPrices, setAllPrices] = React.useState<Record<string, { current: number, diff: number, pct: number }>>({});
  
  React.useEffect(() => {
    if (symbol !== "ALL" || assetClass === "stocks") return;
    
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
  }, [symbol, assetClass]);

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
      <div className="flex flex-wrap items-center gap-2 mb-4">
        {/* Asset-class mode: Crypto / Stocks / Both */}
        <div className="flex gap-1 mr-2 p-0.5 bg-zinc-900 rounded-lg border border-zinc-800">
          {(["crypto", "stocks", "both"] as const).map(m => (
            <button
              key={m}
              onClick={() => setAssetClass(m)}
              className={`px-3 py-1 rounded-md text-xs font-semibold capitalize transition-colors ${assetClass === m ? 'bg-zinc-100 text-zinc-900' : 'text-zinc-400 hover:text-zinc-200'}`}
            >
              {m}
            </button>
          ))}
        </div>
        {["ALL", ...displayedSymbols].map(coin => {
          const att = attentionState[coin];
          const ov = attentionOverrides[coin];
          return (
            <div key={coin} className="flex items-center gap-0.5">
              <button
                onClick={() => setSymbol(coin)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${symbol === coin ? 'bg-zinc-100 text-zinc-900' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'}`}
              >
                {coin.replace('USDT', '')}
              </button>
              {coin !== "ALL" && (
                <button
                  onClick={() => cycleAttention(coin)}
                  title={`Attention: ${att || 'low'}${ov ? ` (override: ${ov})` : ''} — click to cycle auto → HIGH → LOW`}
                  className={`w-2 h-2 rounded-full transition-all ${att === 'high' ? 'bg-red-500' : 'bg-emerald-500/70'} ${ov ? 'ring-2 ring-amber-400/70 ring-offset-1 ring-offset-zinc-950' : 'hover:scale-125'}`}
                />
              )}
            </div>
          );
        })}
      </div>

      <div className="grid gap-6 md:grid-cols-2">

        {symbol === "ALL" ? (
          <Card className="col-span-1 md:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle>All Markets</CardTitle>
              <DollarSign size={15} className="text-zinc-400" />
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-2">
                {displayedSymbols.map(coin => {
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

      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="col-span-1 md:col-span-2 grid gap-6 md:grid-cols-2">
           <VirtualWalletCard viewedSymbol={activeChartSymbol} />
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
            <TradingChart
                symbol={activeChartSymbol}
                currencyRate={rate}
                currencyPrefix={currencySymbol}
                onSymbolChange={(next) => {
                    // Promote the chosen symbol to the dashboard's active selection.
                    // If the picked symbol belongs to the other asset class, also flip the class.
                    if (availableStocks.includes(next) && assetClass !== "stocks") setAssetClass("stocks");
                    else if (availableCoins.includes(next) && assetClass !== "crypto") setAssetClass("crypto");
                    setSymbol(next);
                }}
            />
          </CardContent>
        </Card>

        <Card className="col-span-3 lg:col-span-2 h-full">
          <CardHeader className="border-b border-zinc-800/50 pb-4">
            <CardTitle>Orderbook Alpha</CardTitle>
          </CardHeader>
          <CardContent className="font-mono text-[11px] text-zinc-400 space-y-4 pt-4">
            {isStock(activeChartSymbol) && (
              <div className="text-[10px] text-amber-500/80 -mt-1">
                Live {activeChartSymbol} order book activates once a stock broker is connected (Phase 7b).
              </div>
            )}
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
