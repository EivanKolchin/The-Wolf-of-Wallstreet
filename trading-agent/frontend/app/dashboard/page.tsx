"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import dynamic from 'next/dynamic';
import { useMarketData } from "@/lib/hooks/useMarketData";
import { useNewsData } from "@/lib/hooks/useNewsData";
import { useAppState } from "@/lib/context";
import { StrategyBookCard } from "@/components/StrategyBookCard";
import { BookHealthCard } from "@/components/BookHealthCard";
import { AgentStatusBanner } from "@/components/AgentStatusBanner";
import { NewsScannerWidget } from "@/components/NewsScannerWidget";
import { NewsInsightsWidget } from "@/components/NewsInsightsWidget";
import { MarketAgentChat } from "@/components/MarketAgentChat";
import { MessageCircle, Minimize2 } from "lucide-react";

// Use dynamic import for TradingView widget because it relies on window/document
const TradingChart = dynamic(() => import('@/components/TradingChart'), { ssr: false });

// Fallbacks used until /api/universe responds (kept in sync with backend/core/universe.py).
const FALLBACK_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "RENDERUSDT", "NEARUSDT"];
const FALLBACK_STOCKS = ["SNDK", "AMD", "MU", "BE", "NVDA", "TSM", "SMCI", "TSLA", "MSTR", "COIN", "PLTR"];
const FLOATING_CHAT_BUBBLE = 56;
const FLOATING_CHAT_WIDTH = 420;
const FLOATING_CHAT_HEIGHT = 660;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export default function Dashboard() {
  const [symbol, setSymbol] = React.useState("ALL");
  const [assetClass, setAssetClass] = React.useState<"crypto" | "stocks" | "both">("crypto");
  const [chatMode, setChatMode] = React.useState<"docked" | "floating-minimized" | "floating-open">("docked");
  const [chatPosition, setChatPosition] = React.useState({ x: 0, y: 0 });
  const chatPanelRef = React.useRef<HTMLDivElement>(null);
  const chatDockSlotRef = React.useRef<HTMLDivElement>(null);
  const chatDockRectRef = React.useRef<DOMRect | null>(null);
  const chatDragStateRef = React.useRef({
    dragging: false,
    moved: false,
    offsetX: 0,
    offsetY: 0,
  });

  // Live tradeable universe — pulled from the backend so the tabs always match
  // what the agent actually trades (no more hardcoded, drifting lists).
  const [coins, setCoins] = React.useState<string[]>(FALLBACK_COINS);
  const [stocks, setStocks] = React.useState<string[]>(FALLBACK_STOCKS);
  React.useEffect(() => {
    fetch("http://127.0.0.1:8000/api/universe")
      .then((r) => r.json())
      .then((d) => {
        if (Array.isArray(d?.crypto) && d.crypto.length) setCoins(d.crypto);
        if (Array.isArray(d?.stocks) && d.stocks.length) setStocks(d.stocks);
      })
      .catch(() => { /* keep fallbacks */ });
  }, []);

  const displayedSymbols = assetClass === "crypto" ? coins
    : assetClass === "stocks" ? stocks
    : [...coins, ...stocks];

  // Reset to ALL when switching asset class — unless the current symbol is still
  // valid for the new class (e.g. the chart dropdown flipped the class for us).
  React.useEffect(() => {
    setSymbol(prev => {
      if (prev === "ALL") return prev;
      const validHere = (assetClass === "stocks" && stocks.includes(prev))
                     || (assetClass === "crypto" && coins.includes(prev))
                     || (assetClass === "both");
      return validHere ? prev : "ALL";
    });
  }, [assetClass, coins, stocks]);

  const activeChartSymbol = symbol === "ALL" ? (assetClass === "stocks" ? (stocks[0] || "AMD") : "BTCUSDT") : symbol;

  const { klines } = useMarketData(activeChartSymbol);
  const { rawNews, predictions } = useNewsData();
  const { signals, currency, exchangeRates } = useAppState();

  const rate = exchangeRates[currency] || 1;
  const currencySymbol = currency === 'GBP' ? '£' : currency === 'EUR' ? '€' : currency === 'JPY' ? '¥' : '$';

  // Fetch prices for all coins if "ALL" is selected
  const [allPrices, setAllPrices] = React.useState<Record<string, { current: number, diff: number, pct: number }>>({});

  React.useEffect(() => {
    if (symbol !== "ALL" || assetClass === "stocks") return;

    const fetchStats = async () => {
      try {
        const res = await fetch('https://api.binance.com/api/v3/ticker/24hr?' + new URLSearchParams({
          symbols: JSON.stringify(coins)
        }));
        const data = await res.json();
        const initialPrices: any = {};
        for (const d of data) {
          initialPrices[d.symbol] = { current: parseFloat(d.lastPrice), diff: parseFloat(d.priceChange), pct: parseFloat(d.priceChangePercent) };
        }
        setAllPrices(initialPrices);
      } catch (e) { console.error(e); }
    };
    fetchStats();

    const streams = coins.map(c => `${c.toLowerCase()}@ticker`).join('/');
    const ws = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${streams}`);
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload?.data?.s) {
          const d = payload.data;
          setAllPrices(prev => ({ ...prev, [d.s]: { current: parseFloat(d.c), diff: parseFloat(d.p), pct: parseFloat(d.P) } }));
        }
      } catch (e) { console.error(e); }
    };
    return () => ws.close();
  }, [symbol, assetClass, coins]);

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
  const conviction = latestSignal?.direction || "neutral";
  const convictionColor = conviction === "long" ? "text-emerald-400" : conviction === "short" ? "text-rose-400" : "text-zinc-400";
  const convictionLabel = conviction === "long" ? "Uptrend bias" : conviction === "short" ? "Downtrend bias" : "Sideways / neutral";

  const ensureFloatingChatPosition = React.useCallback(() => {
    setChatPosition((prev) => {
      if (prev.x !== 0 || prev.y !== 0) return prev;
      const x = typeof window !== "undefined" ? window.innerWidth - FLOATING_CHAT_BUBBLE - 24 : 24;
      const y = typeof window !== "undefined" ? window.innerHeight - FLOATING_CHAT_BUBBLE - 24 : 24;
      return {
        x: Math.max(16, x),
        y: Math.max(16, y),
      };
    });
  }, []);

  const captureDockRect = React.useCallback(() => {
    const rect = chatDockSlotRef.current?.getBoundingClientRect();
    if (rect) {
      chatDockRectRef.current = rect;
    }
  }, []);

  const isNearDockTarget = React.useCallback((x: number, y: number) => {
    const rect = chatDockRectRef.current;
    if (!rect) return false;
    const bubbleWidth = chatMode === "floating-open" ? FLOATING_CHAT_WIDTH : FLOATING_CHAT_BUBBLE;
    const bubbleHeight = chatMode === "floating-open" ? FLOATING_CHAT_HEIGHT : FLOATING_CHAT_BUBBLE;
    const bubbleCenterX = x + bubbleWidth / 2;
    const bubbleCenterY = y + bubbleHeight / 2;
    const targetCenterX = rect.left + rect.width / 2;
    const targetCenterY = rect.top + rect.height / 2;
    return Math.abs(bubbleCenterX - targetCenterX) < 140 && Math.abs(bubbleCenterY - targetCenterY) < 140;
  }, [chatMode]);

  React.useEffect(() => {
    if (chatMode === "docked") {
      captureDockRect();
    }
  }, [captureDockRect, chatMode]);

  const beginChatDrag = React.useCallback((event: React.PointerEvent<HTMLDivElement | HTMLButtonElement>) => {
    if (chatMode === "docked") return;
    const panel = chatPanelRef.current;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    chatDragStateRef.current = {
      dragging: true,
      moved: false,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }, [chatMode]);

  const updateChatDrag = React.useCallback((event: React.PointerEvent<HTMLDivElement | HTMLButtonElement>) => {
    if (!chatDragStateRef.current.dragging) return;
    const width = chatMode === "floating-open" ? FLOATING_CHAT_WIDTH : FLOATING_CHAT_BUBBLE;
    const height = chatMode === "floating-open" ? FLOATING_CHAT_HEIGHT : FLOATING_CHAT_BUBBLE;
    const nextX = event.clientX - chatDragStateRef.current.offsetX;
    const nextY = event.clientY - chatDragStateRef.current.offsetY;
    chatDragStateRef.current.moved = chatDragStateRef.current.moved || Math.abs(nextX - chatPosition.x) > 4 || Math.abs(nextY - chatPosition.y) > 4;
    setChatPosition({
      x: clamp(nextX, 12, window.innerWidth - width - 12),
      y: clamp(nextY, 12, window.innerHeight - height - 12),
    });
  }, [chatMode, chatPosition.x, chatPosition.y]);

  const endChatDrag = React.useCallback((event: React.PointerEvent<HTMLDivElement | HTMLButtonElement>) => {
    if (!chatDragStateRef.current.dragging) return;
    chatDragStateRef.current.dragging = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
    if (chatMode !== "docked" && isNearDockTarget(chatPosition.x, chatPosition.y)) {
      chatDragStateRef.current.moved = false;
      setChatMode("docked");
    }
  }, [chatMode, chatPosition.x, chatPosition.y, isNearDockTarget]);

  const minimizeChat = React.useCallback(() => {
    captureDockRect();
    ensureFloatingChatPosition();
    setChatMode("floating-minimized");
  }, [captureDockRect, ensureFloatingChatPosition]);

  const openFloatingChat = React.useCallback(() => {
    if (chatDragStateRef.current.moved) {
      chatDragStateRef.current.moved = false;
      return;
    }
    setChatPosition((prev) => {
      const maxX = typeof window !== "undefined" ? window.innerWidth - FLOATING_CHAT_WIDTH - 24 : prev.x;
      const maxY = typeof window !== "undefined" ? window.innerHeight - FLOATING_CHAT_HEIGHT - 24 : prev.y;
      return {
        x: clamp(prev.x || 24, 12, Math.max(12, maxX)),
        y: clamp(prev.y || 24, 12, Math.max(12, maxY)),
      };
    });
    setChatMode("floating-open");
  }, []);

  const restoreDockedChat = React.useCallback(() => {
    setChatMode("docked");
  }, []);

  return (
    <div className="mx-auto max-w-[1400px] space-y-6 pt-2 font-sans fadeIn">

      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <div className="inline-flex items-center gap-1 rounded-lg border border-[#1a1a1c] bg-[#0a0a0a] p-1">
            {(["crypto", "stocks", "both"] as const).map(m => (
              <button
                key={m}
                onClick={() => setAssetClass(m)}
                className={`rounded-md px-3.5 py-1.5 text-[12px] font-medium capitalize tracking-wide transition-colors ${
                  assetClass === m ? 'bg-zinc-100 text-zinc-900' : 'text-zinc-500 hover:text-zinc-200'
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-[#1a1a1c]" />

          <div className="flex flex-wrap items-center gap-1.5">
            {["ALL", ...displayedSymbols].map(coin => {
              const active = symbol === coin;
              return (
                <button
                  key={coin}
                  onClick={() => setSymbol(coin)}
                  className={`rounded-md px-3 py-1.5 text-[13px] font-medium tracking-wide transition-all ${
                    active
                      ? 'bg-zinc-100 text-zinc-900'
                      : 'border border-[#1a1a1c] bg-[#0e0e10] text-zinc-400 hover:border-[#262629] hover:text-zinc-100'
                  }`}
                >
                  {coin.replace('USDT', '')}
                </button>
              );
            })}
          </div>
        </div>
        <div className="lg:max-w-[360px]">
          <AgentStatusBanner compact />
        </div>
      </div>

      {/* ── Stat row ── */}
      <div className="grid gap-6 md:grid-cols-1">
        {symbol === "ALL" ? (
          <Card className="md:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle>Market Overview</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                {displayedSymbols.map(coin => {
                  const data = allPrices[coin] || { current: 0, diff: 0, pct: 0 };
                  const up = data.diff >= 0;
                  return (
                    <div key={coin} className="rounded-xl border border-[#171717] bg-[#0e0e10] p-3 transition-colors hover:border-[#262629]">
                      <div className="mb-1 text-xs font-medium text-zinc-400">{coin.replace('USDT', '')}</div>
                      <div className="font-mono text-lg font-semibold tracking-tight text-zinc-100">
                        {currencySymbol}{(data.current * rate).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </div>
                      <div className={`mt-1 font-mono text-[11px] ${up ? "text-emerald-400" : "text-rose-400"}`}>
                        {up ? '+' : ''}{(data.diff * rate).toFixed(2)} ({up ? '+' : ''}{data.pct.toFixed(2)}%)
                      </div>
                    </div>
                  );
                })}
                {displayedSymbols.length === 0 && (
                  <div className="col-span-full py-6 text-center text-sm text-zinc-600">No assets in this class.</div>
                )}
              </div>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>{activeChartSymbol.replace('USDT', '')} · Price</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="font-mono text-2xl font-semibold tracking-tight text-zinc-100">
                {currencySymbol}{parsedCurrent.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <p className={`mt-1 font-mono text-[12px] font-medium ${priceDiff >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                {priceDiff >= 0 ? '+' : ''}{priceDiff.toFixed(2)} ({priceDiffPct.toFixed(2)}%)
              </p>
            </CardContent>
          </Card>
        )}

      </div>

      {/* ── Strategy book (managed-beta + TS-momentum): allocations + net-worth curve ── */}
      <div className="grid gap-6 lg:items-start">
        <StrategyBookCard />
      </div>

      {/* ── Book health: realized vol vs target, Sharpe, drawdown, uptime, news/anomaly state ── */}
      <div className="grid gap-6 lg:items-start">
        <BookHealthCard />
      </div>

      {/* ── Chart row ── */}
      <div className="relative min-h-[760px]">
        <Card className="flex min-h-[760px] flex-col overflow-hidden p-0">
          <CardHeader className="flex flex-row items-center justify-between gap-4 border-b border-[#171717] py-4">
            <CardTitle className="text-sm">
              Market Trajectory · {activeChartSymbol.replace('USDT', '')}
            </CardTitle>
            <span className={`rounded-full border border-[#1f1f22] bg-[#0e0e10] px-2.5 py-1 text-[10px] font-medium tracking-wide ${convictionColor}`}>
              {convictionLabel}
            </span>
          </CardHeader>
          <CardContent className="m-0 flex-1 p-0">
            <TradingChart
              symbol={activeChartSymbol}
              currencyRate={rate}
              currencyPrefix={currencySymbol}
              onSymbolChange={(next) => {
                if (stocks.includes(next) && assetClass !== "stocks") setAssetClass("stocks");
                else if (coins.includes(next) && assetClass !== "crypto") setAssetClass("crypto");
                setSymbol(next);
              }}
            />
          </CardContent>
        </Card>
      </div>

      <div className={`grid gap-6 lg:items-start ${chatMode === "docked" ? "lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]" : "lg:grid-cols-1"}`}>
        <div className="min-h-[520px]">
          <NewsScannerWidget rawNews={rawNews} className="h-[520px] min-h-[520px]" />
        </div>

        {chatMode === "docked" && (
          <div ref={chatDockSlotRef} className="relative min-h-[520px]">
            <div className="relative h-full">
              <button
                type="button"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={minimizeChat}
                className="absolute right-3 top-3 z-10 inline-flex h-8 w-8 items-center justify-center rounded-full border border-[#1f1f22] bg-[#0e0e10] text-zinc-500 transition-colors hover:border-zinc-500 hover:text-zinc-200"
                title="Minimize chat"
              >
                <Minimize2 size={14} />
              </button>
              <MarketAgentChat symbol={activeChartSymbol} />
            </div>
          </div>
        )}
      </div>

      {chatMode !== "docked" && (
        <div
          ref={chatPanelRef}
          className={`fixed z-50 ${chatMode === "floating-open" ? "" : "rounded-full"}`}
          style={{
            left: chatPosition.x || undefined,
            top: chatPosition.y || undefined,
            width: chatMode === "floating-open" ? Math.min(FLOATING_CHAT_WIDTH, (typeof window !== "undefined" ? window.innerWidth - 24 : FLOATING_CHAT_WIDTH)) : FLOATING_CHAT_BUBBLE,
          }}
        >
          {chatMode === "floating-open" ? (
            <div className="relative rounded-3xl border border-[#1f1f22] bg-[#09090b] shadow-[0_24px_80px_rgba(0,0,0,0.55)] backdrop-blur">
              <div
                className="flex items-center justify-between rounded-t-3xl border-b border-[#171717] bg-[#0b0b0d] px-4 py-3"
              >
                <div
                  className="flex cursor-move items-center gap-2 text-sm font-medium text-zinc-200"
                  onPointerDown={beginChatDrag}
                  onPointerMove={updateChatDrag}
                  onPointerUp={endChatDrag}
                  onPointerCancel={endChatDrag}
                >
                  <MessageCircle size={15} className="text-emerald-300" />
                  Market Copilot
                </div>
                <button
                  type="button"
                  onPointerDown={(e) => e.stopPropagation()}
                  onClick={minimizeChat}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[#1f1f22] bg-[#0e0e10] text-zinc-500 transition-colors hover:border-zinc-500 hover:text-zinc-200"
                  title="Minimize chat"
                >
                  <Minimize2 size={14} />
                </button>
              </div>
              <div className="p-0">
                <MarketAgentChat symbol={activeChartSymbol} />
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={openFloatingChat}
              onPointerDown={beginChatDrag}
              onPointerMove={updateChatDrag}
              onPointerUp={endChatDrag}
              onPointerCancel={endChatDrag}
              className="flex h-14 w-14 items-center justify-center rounded-full border border-emerald-500/30 bg-[#0b0b0d] text-emerald-300 shadow-[0_20px_60px_rgba(0,0,0,0.45)] transition-transform hover:scale-105"
              title="Open market chat"
            >
              <MessageCircle size={22} />
            </button>
          )}
        </div>
      )}

      <NewsInsightsWidget predictions={predictions} expanded />
    </div>
  );
}
