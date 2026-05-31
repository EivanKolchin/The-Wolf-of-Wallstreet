
"use client";

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, ColorType, IPriceLine, LineStyle } from "lightweight-charts";
import { Search, Pencil, Type, Activity, MousePointer2, Slash, Settings, Trash2, ListMinus, X, Ruler, ArrowRightToLine, Palette, Undo, Redo, Eraser, Plus, Minus, Brain, ChevronDown } from "lucide-react";
import { subscribeToLiveWs, API_BASE } from "@/lib/api";

// Restricted stock universe (mirrors backend/core/universe.py STOCK_UNDERLYINGS).
// Anything else is treated as crypto by the chart's data-source selection.
const STOCK_UNDERLYINGS = ["SNDK", "AMD", "MU", "AXTI", "BE"];
const isStockSymbol = (s: string) => STOCK_UNDERLYINGS.includes((s || "").toUpperCase());

const TIMEFRAMES = [
    { label: "1m", value: "1m" },
    { label: "5m", value: "5m" },
    { label: "15m", value: "15m" },
    { label: "1h", value: "1h" },
    { label: "4h", value: "4h" },
    { label: "1D", value: "1d" },
    { label: "1W", value: "1w" },
    { label: "1M", value: "1M" },
    { label: "3M", value: "3M" },
    { label: "1Y", value: "1Y" },
    { label: "ALL", value: "ALL" }
];

// ===========================================================================
// Phase 5: indicator math. Module-level pure helpers (stable across renders).
// Each takes the chart's candle array [{time, open, high, low, close, volume}]
// and returns lightweight-charts {time, value} (or {time, value, color}) points.
// ===========================================================================
type Candle = { time: any; open: number; high: number; low: number; close: number; volume?: number };

const _ema = (vals: number[], period: number): number[] => {
    const k = 2 / (period + 1);
    const out: number[] = [];
    let e = vals[0] ?? 0;
    for (let i = 0; i < vals.length; i++) { e = i === 0 ? vals[0] : (vals[i] - e) * k + e; out.push(e); }
    return out;
};
const _sma = (vals: number[], period: number): (number | null)[] =>
    vals.map((_, i) => i < period - 1 ? null : vals.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period);

const calcRSI = (data: Candle[], period = 14) => {
    const out: { time: any; value: number }[] = [];
    let gain = 0, loss = 0;
    for (let i = 1; i < data.length; i++) {
        const ch = data[i].close - data[i - 1].close;
        const g = Math.max(ch, 0), l = Math.max(-ch, 0);
        if (i <= period) { gain += g; loss += l; if (i === period) { const rs = loss === 0 ? 100 : gain / loss; out.push({ time: data[i].time, value: 100 - 100 / (1 + rs) }); } }
        else { gain = (gain * (period - 1) + g) / period; loss = (loss * (period - 1) + l) / period; const rs = loss === 0 ? 100 : gain / loss; out.push({ time: data[i].time, value: 100 - 100 / (1 + rs) }); }
    }
    return out;
};
const calcMACD = (data: Candle[], fast = 12, slow = 26, signal = 9) => {
    const closes = data.map(d => d.close);
    const ef = _ema(closes, fast), es = _ema(closes, slow);
    const macd = closes.map((_, i) => ef[i] - es[i]);
    const sig = _ema(macd, signal);
    const macdLine = data.map((d, i) => ({ time: d.time, value: macd[i] }));
    const signalLine = data.map((d, i) => ({ time: d.time, value: sig[i] }));
    const hist = data.map((d, i) => ({ time: d.time, value: macd[i] - sig[i], color: (macd[i] - sig[i]) >= 0 ? 'rgba(52,211,153,0.6)' : 'rgba(248,113,113,0.6)' }));
    return { macdLine, signalLine, hist };
};
const calcStochastic = (data: Candle[], kP = 14, dP = 3) => {
    const k: { time: any; value: number }[] = [];
    for (let i = kP - 1; i < data.length; i++) {
        const slice = data.slice(i - kP + 1, i + 1);
        const hi = Math.max(...slice.map(s => s.high)), lo = Math.min(...slice.map(s => s.low));
        k.push({ time: data[i].time, value: hi === lo ? 50 : ((data[i].close - lo) / (hi - lo)) * 100 });
    }
    const dVals = _sma(k.map(x => x.value), dP);
    const d = k.map((x, i) => ({ time: x.time, value: dVals[i] })).filter(x => x.value !== null) as { time: any; value: number }[];
    return { k, d };
};
const calcCCI = (data: Candle[], period = 20) => {
    const out: { time: any; value: number }[] = [];
    for (let i = period - 1; i < data.length; i++) {
        const tp = data.slice(i - period + 1, i + 1).map(s => (s.high + s.low + s.close) / 3);
        const ma = tp.reduce((a, b) => a + b, 0) / period;
        const md = tp.reduce((a, b) => a + Math.abs(b - ma), 0) / period;
        const cur = (data[i].high + data[i].low + data[i].close) / 3;
        out.push({ time: data[i].time, value: md === 0 ? 0 : (cur - ma) / (0.015 * md) });
    }
    return out;
};
const calcAO = (data: Candle[]) => {
    const mp = data.map(d => (d.high + d.low) / 2);
    const s5 = _sma(mp, 5), s34 = _sma(mp, 34);
    return data.map((d, i) => (s5[i] === null || s34[i] === null) ? null : ({
        time: d.time, value: (s5[i] as number) - (s34[i] as number),
        color: i > 0 && ((s5[i] as number) - (s34[i] as number)) >= ((s5[i - 1] as number) - (s34[i - 1] as number)) ? 'rgba(52,211,153,0.6)' : 'rgba(248,113,113,0.6)',
    })).filter(Boolean) as any[];
};
const calcMomentum = (data: Candle[], period = 10) =>
    data.map((d, i) => i < period ? null : ({ time: d.time, value: d.close - data[i - period].close })).filter(Boolean) as any[];
const calcBollinger = (data: Candle[], period = 20, mult = 2) => {
    const closes = data.map(d => d.close);
    const mid = _sma(closes, period);
    const upper: any[] = [], lower: any[] = [], middle: any[] = [];
    for (let i = 0; i < data.length; i++) {
        if (mid[i] === null) continue;
        const slice = closes.slice(i - period + 1, i + 1);
        const m = mid[i] as number;
        const sd = Math.sqrt(slice.reduce((a, b) => a + (b - m) ** 2, 0) / period);
        upper.push({ time: data[i].time, value: m + mult * sd });
        lower.push({ time: data[i].time, value: m - mult * sd });
        middle.push({ time: data[i].time, value: m });
    }
    return { upper, middle, lower };
};
const calcATR = (data: Candle[], period = 14) => {
    const tr: number[] = [];
    for (let i = 0; i < data.length; i++) {
        if (i === 0) { tr.push(data[i].high - data[i].low); continue; }
        tr.push(Math.max(data[i].high - data[i].low, Math.abs(data[i].high - data[i - 1].close), Math.abs(data[i].low - data[i - 1].close)));
    }
    const atr = _ema(tr, period);
    return data.map((d, i) => ({ time: d.time, value: atr[i] })).slice(period);
};
const calcVWAP = (data: Candle[]) => {
    let cumPV = 0, cumV = 0;
    return data.map(d => { const tp = (d.high + d.low + d.close) / 3; const v = d.volume || 0; cumPV += tp * v; cumV += v; return { time: d.time, value: cumV === 0 ? d.close : cumPV / cumV }; });
};
const calcOBV = (data: Candle[]) => {
    let obv = 0;
    return data.map((d, i) => { if (i > 0) { if (d.close > data[i - 1].close) obv += (d.volume || 0); else if (d.close < data[i - 1].close) obv -= (d.volume || 0); } return { time: d.time, value: obv }; });
};
const calcCMF = (data: Candle[], period = 20) => {
    const out: { time: any; value: number }[] = [];
    for (let i = period - 1; i < data.length; i++) {
        const slice = data.slice(i - period + 1, i + 1);
        let mfv = 0, vol = 0;
        for (const s of slice) { const range = s.high - s.low; const mult = range === 0 ? 0 : ((s.close - s.low) - (s.high - s.close)) / range; mfv += mult * (s.volume || 0); vol += (s.volume || 0); }
        out.push({ time: data[i].time, value: vol === 0 ? 0 : mfv / vol });
    }
    return out;
};
const calcVolume = (data: Candle[]) =>
    data.map(d => ({ time: d.time, value: d.volume || 0, color: d.close >= d.open ? 'rgba(52,211,153,0.5)' : 'rgba(248,113,113,0.5)' }));
const calcIchimoku = (data: Candle[]) => {
    const hh = (s: Candle[]) => Math.max(...s.map(x => x.high));
    const ll = (s: Candle[]) => Math.min(...s.map(x => x.low));
    const line = (period: number) => data.map((d, i) => i < period - 1 ? null : ({ time: d.time, value: (hh(data.slice(i - period + 1, i + 1)) + ll(data.slice(i - period + 1, i + 1))) / 2 }));
    const tenkan = line(9), kijun = line(26);
    const spanA = data.map((d, i) => (tenkan[i] && kijun[i]) ? ({ time: d.time, value: ((tenkan[i] as any).value + (kijun[i] as any).value) / 2 }) : null);
    const spanB = line(52);
    const clean = (a: any[]) => a.filter(Boolean);
    return { tenkan: clean(tenkan), kijun: clean(kijun), spanA: clean(spanA), spanB: clean(spanB) };
};

export default function TradingChart({
    symbol = "BTCUSDT",
    currencyRate = 1,
    currencyPrefix = "$",
    onSymbolChange,
}: {
    symbol?: string,
    currencyRate?: number,
    currencyPrefix?: string,
    onSymbolChange?: (next: string) => void,
}) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const predictionSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    // Phase 18 v1: explicit median line on top of a shaded p25/p75 band.
    // Cycle 22.1: the band is two stacked Area series (p75 violet fill + p25 mask),
    // so these refs are Area, not Line.
    const predictionMedianRef = useRef<ISeriesApi<"Line"> | null>(null);
    const predictionP25Ref = useRef<ISeriesApi<"Area"> | null>(null);
    const predictionP75Ref = useRef<ISeriesApi<"Area"> | null>(null);
    const buyPriceLineRef = useRef<IPriceLine | null>(null);
    const sellPriceLineRef = useRef<IPriceLine | null>(null);
    const ema9Ref = useRef<ISeriesApi<"Line"> | null>(null);
    const ema21Ref = useRef<ISeriesApi<"Line"> | null>(null);
    const sma50Ref = useRef<ISeriesApi<"Line"> | null>(null);
    const sma200Ref = useRef<ISeriesApi<"Line"> | null>(null);
    const vwmaRef = useRef<ISeriesApi<"Line"> | null>(null);

    const [chartData, setChartData] = useState<any[]>([]);
        const [timeframe, setTimeframe] = useState("15m");
    const [cursorDate, setCursorDate] = useState<number | null>(null);

    // Symbol-picker dropdown (clickable label → asset list).
    const [showSymbolDrop, setShowSymbolDrop] = useState(false);
    const [symbolSearch, setSymbolSearch] = useState("");  // Cycle 22.2 filter
    const [universe, setUniverse] = useState<{ crypto: string[], stocks: string[] }>({ crypto: [], stocks: [] });
    const [dataError, setDataError] = useState<string | null>(null);
    const symbolDropRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        let cancelled = false;
        fetch(`${API_BASE}/market/universe`)
            .then(r => r.json())
            .then((u: any) => {
                if (cancelled) return;
                setUniverse({
                    crypto: Array.isArray(u?.crypto) ? u.crypto : [],
                    stocks: Array.isArray(u?.stocks) ? u.stocks : [],
                });
            })
            .catch(() => {
                // Fall back to the static list so the dropdown still works offline.
                setUniverse({
                    crypto: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AAVEUSDT", "XLMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"],
                    stocks: STOCK_UNDERLYINGS,
                });
            });
        const onDocClick = (e: MouseEvent) => {
            if (symbolDropRef.current && !symbolDropRef.current.contains(e.target as Node)) {
                setShowSymbolDrop(false);
                setSymbolSearch("");
            }
        };
        // Cycle 22.2: Esc closes the dropdown — matches the muscle-memory of
        // most chart UIs and means keyboard users aren't trapped.
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                setShowSymbolDrop(false);
                setSymbolSearch("");
            }
        };
        document.addEventListener("mousedown", onDocClick);
        document.addEventListener("keydown", onKeyDown);
        return () => {
            cancelled = true;
            document.removeEventListener("mousedown", onDocClick);
            document.removeEventListener("keydown", onKeyDown);
        };
    }, []);

    // Scroll Pagination States
    const isFetchingHistory = useRef(false);
    const hasMoreHistory = useRef(true);
    const prevRangeInfo = useRef<any>(null);
    const chartDataRef = useRef<any[]>([]);
    const isSwitchingTfRef = useRef(false);

    // UI Panels
    const [showDatePanel, setShowDatePanel] = useState(false);
    const [showFibMenu, setShowFibMenu] = useState(false);
    const [showIndicators, setShowIndicators] = useState(false);
    const [showPredictions, setShowPredictions] = useState(true);
    const [showSettings, setShowSettings] = useState(false);
    
    // Date states
    const [optYear, setOptYear] = useState("");
    const [optMonth, setOptMonth] = useState("");
    const [optDay, setOptDay] = useState("");

    // Customization Settings
    const [config, setConfig] = useState({
        ema9: { show: true, color: '#38BDF8', lineWidth: 1 },
        ema21: { show: true, color: '#FCD34D', lineWidth: 2 },
        sma50: { show: false, color: '#A78BFA', lineWidth: 2 },
        sma200: { show: false, color: '#F43F5E', lineWidth: 2 },
        vwma: { show: false, color: '#10B981', lineWidth: 2 },
        fibColor: '#a78bfa',
        fibOpacity: 0.3,
        drawingColor: '#a78bfa'
    });
    
    const [isVibrantColors, setIsVibrantColors] = useState(false);

    // Drawing Engine States
    const [activeTool, setActiveTool] = useState("pointer");
    const [drawings, setDrawings] = useState<any[]>([]);
    const [undoStack, setUndoStack] = useState<any[][]>([]);
    const [redoStack, setRedoStack] = useState<any[][]>([]);
    
    const [isDrawing, setIsDrawing] = useState(false);
    const [currentDrawing, setCurrentDrawing] = useState<any>(null);
    const [dragCtx, setDragCtx] = useState<{ id: number, type: 'move'|'p1'|'p2'|'rotate', initMouseX: number, initMouseY: number, startDrawing: any } | null>(null);
    const [selectedDrawing, setSelectedDrawing] = useState<number | null>(null);
    const [renderTick, setRenderTick] = useState(0);
    const [expandedMeasure, setExpandedMeasure] = useState<number | null>(null);
    const [lastPredictionData, setLastPredictionData] = useState<any>(null);

    // Phase 5: toggle state for all the previously-stub indicators. Keyed by
    // indicator id; a single effect (below) renders/destroys their series.
    const [indi, setIndi] = useState<Record<string, boolean>>({
        rsi: false, macd: false, stochastic: false, cci: false, ao: false, momentum: false,
        bollinger: false, atr: false, volume: false, vwap: false, obv: false, cmf: false, ichimoku: false,
    });
    const indiSeriesRef = useRef<Record<string, any[]>>({});

    const commitDrawingUpdate = (newDrawings: any[]) => {
        setUndoStack(prev => [...prev, drawings]);
        setRedoStack([]);
        setDrawings(newDrawings);
    };

    const undoDrawing = () => {
        if (undoStack.length > 0) {
            const prev = undoStack[undoStack.length - 1];
            setRedoStack(prevRedo => [...prevRedo, drawings]);
            setDrawings(prev);
            setUndoStack(prevUndo => prevUndo.slice(0, -1));
        }
    };

    const redoDrawing = () => {
        if (redoStack.length > 0) {
            const next = redoStack[redoStack.length - 1];
            setUndoStack(prevUndo => [...prevUndo, drawings]);
            setDrawings(next);
            setRedoStack(prevRedo => prevRedo.slice(0, -1));
        }
    };

        // WebSocket Live Updates (crypto) — stocks fall through to polling below.
    useEffect(() => {
        if (cursorDate || !symbol) return; // Do not live update historical views
        let fetchTimeframe = timeframe;
        if (timeframe === '3M' || timeframe === '1Y' || timeframe === 'ALL') fetchTimeframe = '1M';

        // Stocks: connect to the backend's Alpaca WS proxy for per-trade ticks
        // (smooth, no polling). Backend holds the Alpaca auth + multiplexes.
        if (isStockSymbol(symbol)) {
            // Derive the WS URL from API_BASE so it works on whatever host the
            // backend is bound to (and switches scheme if API_BASE ever becomes https).
            const apiUrl = new URL(API_BASE);
            const wsProto = apiUrl.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${wsProto}//${apiUrl.host}/ws/stocks?symbol=${encodeURIComponent(symbol)}`;
            let reconnect = true;
            let ws: WebSocket | null = null;
            let retryMs = 1000;

            // Trades push high/low + close (canonical price prints). Quotes
            // only nudge close (mid-price) so the line moves smoothly between
            // sparse trade prints without distorting the candle's wick.
            //
            // Cycle 22.5: cap quote-driven repaints to ~20 Hz (50 ms) — fast
            // enough to look continuous on a 60 Hz monitor, slow enough to
            // dodge the chatter when a stock prints 30+ quotes per second.
            // Trade ticks bypass the throttle (every trade is canonical).
            const lastQuotePaintMs = { current: 0 };
            const tickCandle = (tickPrice: number, kind: "trade" | "quote") => {
                if (!chartDataRef.current || chartDataRef.current.length === 0) return;
                if (kind === "quote") {
                    const now = performance.now();
                    if (now - lastQuotePaintMs.current < 50) return;
                    lastQuotePaintMs.current = now;
                }
                const lastIdx = chartDataRef.current.length - 1;
                const last = chartDataRef.current[lastIdx];
                const px = tickPrice * currencyRate;
                const updated = (kind === "trade")
                    ? { ...last, high: Math.max(last.high, px), low: Math.min(last.low, px), close: px }
                    : { ...last, close: px };
                chartDataRef.current[lastIdx] = updated;
                if (seriesRef.current) seriesRef.current.update(updated as any);
            };

            const open = () => {
                if (!reconnect) return;
                ws = new WebSocket(wsUrl);
                ws.onopen = () => { retryMs = 1000; };
                ws.onmessage = (ev) => {
                    try {
                        const m = JSON.parse(ev.data);
                        if (m?.type === "trade" && typeof m.price === "number" && m.price > 0) {
                            tickCandle(m.price, "trade");
                        } else if (m?.type === "quote" && typeof m.mid === "number" && m.mid > 0) {
                            tickCandle(m.mid, "quote");
                        }
                    } catch { /* ignore */ }
                };
                ws.onclose = () => {
                    if (!reconnect) return;
                    setTimeout(open, retryMs);
                    retryMs = Math.min(retryMs * 2, 15000);
                };
                ws.onerror = () => { try { ws?.close(); } catch {} };
            };
            open();
            return () => { reconnect = false; try { ws?.close(); } catch {} };
        }

        const safeSymbol = symbol.toLowerCase().replace(/[^a-z0-9]/g, '');
        const wsUrl = `wss://stream.binance.com:9443/stream?streams=${safeSymbol}@kline_${fetchTimeframe.toLowerCase()}/${safeSymbol}@aggTrade`;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const payload = JSON.parse(event.data);
            if (!payload?.data) return;
            const message = payload.data;

            if (message.e === 'aggTrade') {
                if (!chartDataRef.current || chartDataRef.current.length === 0) return;
                const lastIdx = chartDataRef.current.length - 1;
                const lastCandle = chartDataRef.current[lastIdx];
                const tickPrice = parseFloat(message.p) * currencyRate;
                const updatedCandle = {
                    ...lastCandle,
                    high: Math.max(lastCandle.high, tickPrice),
                    low: Math.min(lastCandle.low, tickPrice),
                    close: tickPrice,
                };
                chartDataRef.current[lastIdx] = updatedCandle;
                if (seriesRef.current) {
                    seriesRef.current.update(updatedCandle as any);
                }
                if (config.ema9.show && ema9Ref.current) {
                     const emaTails = calculateEMA(chartDataRef.current, 9);
                     if (emaTails.length > 0) ema9Ref.current.update(emaTails[emaTails.length - 1] as any);
                }
                if (config.ema21.show && ema21Ref.current) {
                     const emaTails = calculateEMA(chartDataRef.current, 21);
                     if (emaTails.length > 0) ema21Ref.current.update(emaTails[emaTails.length - 1] as any);
                }
                if (config.sma50?.show && sma50Ref.current) {
                     const smaTails = calculateSMA(chartDataRef.current, 50);
                     if (smaTails.length > 0) sma50Ref.current.update(smaTails[smaTails.length - 1] as any);
                }
                if (config.sma200?.show && sma200Ref.current) {
                     const smaTails = calculateSMA(chartDataRef.current, 200);
                     if (smaTails.length > 0) sma200Ref.current.update(smaTails[smaTails.length - 1] as any);
                }
                if (config.vwma?.show && vwmaRef.current) {
                     const vwmaTails = calculateVWMA(chartDataRef.current, 20);
                     if (vwmaTails.length > 0) vwmaRef.current.update(vwmaTails[vwmaTails.length - 1] as any);
                }
                return;
            }

            if (message.e === 'kline') {
                const k = message.k;
                const newCandle = {
                    time: k.t / 1000,
                    open: parseFloat(k.o) * currencyRate,
                    high: parseFloat(k.h) * currencyRate,
                    low: parseFloat(k.l) * currencyRate,
                    close: parseFloat(k.c) * currencyRate,
                };

                if (seriesRef.current) {
                    seriesRef.current.update(newCandle as any);
                }

                if (chartDataRef.current && chartDataRef.current.length > 0) {
                    const lastIdx = chartDataRef.current.length - 1;
                    const lastCandle = chartDataRef.current[lastIdx];
                    if (newCandle.time === lastCandle.time) {
                        chartDataRef.current[lastIdx] = newCandle;
                    } else if (newCandle.time > lastCandle.time) {
                        chartDataRef.current.push(newCandle);
                    }

                    // Dynamically recalculate EMA/SMA tails
                    if (config.ema9.show && ema9Ref.current) {
                         const emaTails = calculateEMA(chartDataRef.current, 9);
                         if (emaTails.length > 0) ema9Ref.current.update(emaTails[emaTails.length - 1] as any);
                    }
                    if (config.ema21.show && ema21Ref.current) {
                         const emaTails = calculateEMA(chartDataRef.current, 21);
                         if (emaTails.length > 0) ema21Ref.current.update(emaTails[emaTails.length - 1] as any);
                    }
                    if (config.sma50?.show && sma50Ref.current) {
                         const smaTails = calculateSMA(chartDataRef.current, 50);
                         if (smaTails.length > 0) sma50Ref.current.update(smaTails[smaTails.length - 1] as any);
                    }
                    if (config.sma200?.show && sma200Ref.current) {
                         const smaTails = calculateSMA(chartDataRef.current, 200);
                         if (smaTails.length > 0) sma200Ref.current.update(smaTails[smaTails.length - 1] as any);
                    }
                    if (config.vwma?.show && vwmaRef.current) {
                         const vwmaTails = calculateVWMA(chartDataRef.current, 20);
                         if (vwmaTails.length > 0) vwmaRef.current.update(vwmaTails[vwmaTails.length - 1] as any);
                    }
                }
            }
        };

        return () => ws.close();
    }, [symbol, timeframe, cursorDate, config.ema9.show, config.ema21.show, currencyRate]);

    // Backend Live Updates for Predictions
    useEffect(() => {
        // Phase 1 bug fix: on symbol switch, clear the previous asset's
        // prediction so we never paint stale BTC candles over ETH/AMD/etc.
        // The render effect below also guards on data.symbol === symbol,
        // so a late-arriving stale WS tick can't bring it back.
        setLastPredictionData(null);
        if (predictionSeriesRef.current) predictionSeriesRef.current.setData([]);
        if (predictionMedianRef.current) predictionMedianRef.current.setData([]);
        if (predictionP25Ref.current) predictionP25Ref.current.setData([]);
        if (predictionP75Ref.current) predictionP75Ref.current.setData([]);
        if (seriesRef.current) {
            if (buyPriceLineRef.current) seriesRef.current.removePriceLine(buyPriceLineRef.current);
            if (sellPriceLineRef.current) seriesRef.current.removePriceLine(sellPriceLineRef.current);
            buyPriceLineRef.current = null;
            sellPriceLineRef.current = null;
        }

        // Prime the band from the per-symbol REST endpoint so the user sees
        // the correct asset's prediction immediately, not after waiting up to
        // ~5s for the next WS broadcast tick.
        let cancelled = false;
        (async () => {
            try {
                const res = await fetch(`${API_BASE}/predictions/${encodeURIComponent(symbol)}`);
                if (!res.ok) return;
                const payload = await res.json();
                if (cancelled) return;
                if (payload?.symbol === symbol && Array.isArray(payload?.predictions) && payload.predictions.length > 0) {
                    setLastPredictionData(payload);
                }
            } catch { /* silent — WS will deliver the next tick */ }
        })();

        const liveWs = subscribeToLiveWs((topic, data) => {
            if (topic === "prediction_update") {
                 if (data?.symbol === symbol) {
                    setLastPredictionData(data);
                 }
            }
        });

        return () => { cancelled = true; liveWs.close(); };
    }, [symbol]);

    useEffect(() => {
        if (!lastPredictionData || !predictionSeriesRef.current) return;
        const data = lastPredictionData;
        // Phase 1 bug fix: never paint a band whose symbol doesn't match the
        // currently-viewed asset. Defends against late stale WS ticks landing
        // after a switch.
        if (data?.symbol && data.symbol !== symbol) return;

        if (showPredictions) {
            // Ensure chartData exists to derive timestamps
            if (!chartDataRef.current || chartDataRef.current.length === 0) return;
            
            const lastCandle = chartDataRef.current[chartDataRef.current.length - 1];
            const baseTime = typeof lastCandle.time === "string" ? new Date(lastCandle.time).getTime() / 1000 : lastCandle.time;
            
            let seconds = 900;
            const tf = timeframe.toLowerCase();
            if (tf === "1m") seconds = 60;
            else if (tf === "5m") seconds = 300;
            else if (tf === "15m") seconds = 900;
            else if (tf === "1h") seconds = 3600;
            else if (tf === "4h") seconds = 14400;
            else if (tf === "1d") seconds = 86400;
            else if (tf === "1w") seconds = 604800;
            else if (tf === "1m") seconds = 2592000;
            
            const predsList = data.predictions || [];
            const preds = predsList.map((p: any) => {
                const step = p.step || 1;
                const opacity = Math.max(0.15, 0.8 - (step * (0.65 / 12)));
                const wickColor = `rgba(167, 139, 250, ${Math.min(1, opacity + 0.2)})`;
                return {
                    time: (baseTime + (step * seconds)) as import('lightweight-charts').Time,
                    open: p.open * currencyRate,
                    high: p.high * currencyRate,
                    low: p.low * currencyRate,
                    close: p.close * currencyRate,
                    color: `rgba(167, 139, 250, ${opacity})`,
                    borderColor: wickColor,
                    wickColor: wickColor
                };
            });

            predictionSeriesRef.current.setData(preds);

            // Phase 18 v1: median + p25/p75 lines (fall back to candle close if the
            // backend hasn't been updated yet to emit the explicit band fields).
            const medianLine = predsList.map((p: any) => ({
                time: (baseTime + ((p.step || 1) * seconds)) as import('lightweight-charts').Time,
                value: (p.median_close ?? p.close) * currencyRate,
            }));
            const p25Line = predsList.map((p: any) => ({
                time: (baseTime + ((p.step || 1) * seconds)) as import('lightweight-charts').Time,
                value: (p.p25_close ?? p.low) * currencyRate,
            }));
            const p75Line = predsList.map((p: any) => ({
                time: (baseTime + ((p.step || 1) * seconds)) as import('lightweight-charts').Time,
                value: (p.p75_close ?? p.high) * currencyRate,
            }));
            if (predictionMedianRef.current) predictionMedianRef.current.setData(medianLine);
            if (predictionP25Ref.current)    predictionP25Ref.current.setData(p25Line);
            if (predictionP75Ref.current)    predictionP75Ref.current.setData(p75Line);

            if (seriesRef.current) {
                if (buyPriceLineRef.current) seriesRef.current.removePriceLine(buyPriceLineRef.current);
                if (sellPriceLineRef.current) seriesRef.current.removePriceLine(sellPriceLineRef.current);
                buyPriceLineRef.current = null;
                sellPriceLineRef.current = null;

                if (data.target_buy_price) {
                    buyPriceLineRef.current = seriesRef.current.createPriceLine({
                        price: data.target_buy_price * currencyRate,
                        color: '#006400',
                        lineWidth: 2,
                        lineStyle: LineStyle.Dashed,
                        axisLabelVisible: true,
                        title: 'BUY TARGET      ',
                    });
                }
                if (data.target_sell_price) {
                    sellPriceLineRef.current = seriesRef.current.createPriceLine({
                        price: data.target_sell_price * currencyRate,
                        color: '#800020',
                        lineWidth: 2,
                        lineStyle: LineStyle.Dashed,
                        axisLabelVisible: true,
                        title: 'SELL TARGET      ',
                    });
                }
            }
        } else if (!showPredictions && predictionSeriesRef.current) {
             predictionSeriesRef.current.setData([]);
             if (predictionMedianRef.current) predictionMedianRef.current.setData([]);
             if (predictionP25Ref.current) predictionP25Ref.current.setData([]);
             if (predictionP75Ref.current) predictionP75Ref.current.setData([]);
             if (seriesRef.current) {
                if (buyPriceLineRef.current) seriesRef.current.removePriceLine(buyPriceLineRef.current);
                if (sellPriceLineRef.current) seriesRef.current.removePriceLine(sellPriceLineRef.current);
                buyPriceLineRef.current = null;
                sellPriceLineRef.current = null;
             }
        }
    }, [lastPredictionData, showPredictions, timeframe, currencyRate, symbol]);

    // Phase 5: render/destroy all the toggleable indicators. Overlays (Bollinger,
    // VWAP, Ichimoku) sit on the price scale; oscillators/volume get stacked,
    // non-overlapping bands in the bottom region via dedicated price scales.
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart) return;
        const data = chartData;
        // Tear down any previously-created indicator series first.
        Object.values(indiSeriesRef.current).flat().forEach((s) => {
            try { chart.removeSeries(s); } catch { /* already gone */ }
        });
        indiSeriesRef.current = {};
        if (!data || data.length === 0) return;

        const add = (key: string, s: any) => { (indiSeriesRef.current[key] ||= []).push(s); };

        // ---- overlays on the main price scale ----
        if (indi.bollinger) {
            const bb = calcBollinger(data);
            const u = chart.addLineSeries({ color: 'rgba(96,165,250,0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'BB up' });
            const m = chart.addLineSeries({ color: 'rgba(96,165,250,0.4)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'BB mid' });
            const l = chart.addLineSeries({ color: 'rgba(96,165,250,0.7)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'BB low' });
            u.setData(bb.upper); m.setData(bb.middle); l.setData(bb.lower);
            add('bollinger', u); add('bollinger', m); add('bollinger', l);
        }
        if (indi.vwap) {
            const s = chart.addLineSeries({ color: '#f59e0b', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, title: 'VWAP' });
            s.setData(calcVWAP(data)); add('vwap', s);
        }
        if (indi.ichimoku) {
            const ich = calcIchimoku(data);
            const t = chart.addLineSeries({ color: '#ef4444', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'Tenkan' });
            const k = chart.addLineSeries({ color: '#3b82f6', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'Kijun' });
            const a = chart.addLineSeries({ color: 'rgba(34,197,94,0.6)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'Span A' });
            const b = chart.addLineSeries({ color: 'rgba(168,85,247,0.6)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false, title: 'Span B' });
            t.setData(ich.tenkan); k.setData(ich.kijun); a.setData(ich.spanA); b.setData(ich.spanB);
            add('ichimoku', t); add('ichimoku', k); add('ichimoku', a); add('ichimoku', b);
        }

        // ---- stacked bottom bands for oscillators / volume ----
        const bottomList = ['volume', 'rsi', 'macd', 'stochastic', 'cci', 'ao', 'momentum', 'atr', 'obv', 'cmf'].filter(k => indi[k]);
        const bandTop = 0.72;                       // bottom 28% reserved for indicators
        const bandH = bottomList.length > 0 ? (1 - bandTop) / bottomList.length : 0;
        // Compress the main price scale so candles don't overlap the bands.
        try {
            chart.priceScale('right').applyOptions({
                scaleMargins: bottomList.length > 0 ? { top: 0.05, bottom: 1 - bandTop + 0.02 } : { top: 0.1, bottom: 0.1 },
            });
        } catch { /* noop */ }

        bottomList.forEach((key, slot) => {
            const top = bandTop + slot * bandH;
            const margins = { top, bottom: Math.max(0, 1 - (top + bandH)) };
            const scaleId = `ind_${key}`;
            const lineOpts = (color: string, title: string) => ({ priceScaleId: scaleId, color, lineWidth: 1 as any, priceLineVisible: false, lastValueVisible: false, title });
            const histOpts = { priceScaleId: scaleId, priceFormat: { type: 'volume' as const }, priceLineVisible: false, lastValueVisible: false };
            if (key === 'volume') { const s = chart.addHistogramSeries(histOpts); s.setData(calcVolume(data)); add(key, s); }
            else if (key === 'rsi') { const s = chart.addLineSeries(lineOpts('#a78bfa', 'RSI')); s.setData(calcRSI(data)); add(key, s); }
            else if (key === 'macd') { const m = calcMACD(data); const h = chart.addHistogramSeries({ priceScaleId: scaleId, priceLineVisible: false, lastValueVisible: false }); h.setData(m.hist); const ml = chart.addLineSeries(lineOpts('#3b82f6', 'MACD')); ml.setData(m.macdLine); const sl = chart.addLineSeries(lineOpts('#f59e0b', 'Signal')); sl.setData(m.signalLine); add(key, h); add(key, ml); add(key, sl); }
            else if (key === 'stochastic') { const st = calcStochastic(data); const sk = chart.addLineSeries(lineOpts('#22d3ee', '%K')); sk.setData(st.k); const sd = chart.addLineSeries(lineOpts('#f472b6', '%D')); sd.setData(st.d); add(key, sk); add(key, sd); }
            else if (key === 'cci') { const s = chart.addLineSeries(lineOpts('#eab308', 'CCI')); s.setData(calcCCI(data)); add(key, s); }
            else if (key === 'ao') { const s = chart.addHistogramSeries({ priceScaleId: scaleId, priceLineVisible: false, lastValueVisible: false }); s.setData(calcAO(data)); add(key, s); }
            else if (key === 'momentum') { const s = chart.addLineSeries(lineOpts('#34d399', 'Momentum')); s.setData(calcMomentum(data)); add(key, s); }
            else if (key === 'atr') { const s = chart.addLineSeries(lineOpts('#fb923c', 'ATR')); s.setData(calcATR(data)); add(key, s); }
            else if (key === 'obv') { const s = chart.addLineSeries(lineOpts('#94a3b8', 'OBV')); s.setData(calcOBV(data)); add(key, s); }
            else if (key === 'cmf') { const s = chart.addLineSeries(lineOpts('#c084fc', 'CMF')); s.setData(calcCMF(data)); add(key, s); }
            try { chart.priceScale(scaleId).applyOptions({ scaleMargins: margins }); } catch { /* noop */ }
        });
    }, [indi, chartData, symbol]);

    // Fetch Data — route via the backend so stocks (Alpaca) and crypto (Binance)
    // both work without the frontend having to know which provider serves what.
    useEffect(() => {
        hasMoreHistory.current = true;
        isFetchingHistory.current = false;
        let fetchTimeframe = timeframe;
        if (timeframe === "3M" || timeframe === "1Y" || timeframe === "ALL") fetchTimeframe = "1M";

        const params = new URLSearchParams({
            symbol, interval: fetchTimeframe, limit: "1000",
        });
        if (cursorDate) params.set("startTime", String(cursorDate));
        const url = `${API_BASE}/market/klines?${params.toString()}`;

        fetch(url)
            .then(res => res.json())
            .then((data: any) => {
                // Backend may return either an array (klines) or {error, bars: []}
                const rows: any[] = Array.isArray(data) ? data : (data?.bars || []);
                if (!Array.isArray(data) && data?.error) {
                    const detail = typeof data.detail === "string" ? data.detail
                                   : data.detail?.message || JSON.stringify(data.detail || {});
                    setDataError(`${data.error}${data.status ? ` (${data.status})` : ""}${detail ? `: ${detail}` : ""}`);
                } else if (rows.length === 0) {
                    setDataError("No data returned. If this is a stock, market may be closed and the IEX free feed has a 15-min delay.");
                } else {
                    setDataError(null);
                }
                const formatted = rows.map((d: any) => ({
                    time: d[0] / 1000,
                    open: parseFloat(d[1]) * currencyRate,
                    high: parseFloat(d[2]) * currencyRate,
                    low: parseFloat(d[3]) * currencyRate,
                    close: parseFloat(d[4]) * currencyRate,
                    volume: d[5] ? parseFloat(d[5]) : 0,
                })).filter((c: any) => Number.isFinite(c.time) && Number.isFinite(c.close) && c.close > 0);
                const uniqueData = formatted.filter((v: any, i: number, a: any[]) => a.findIndex((t: any) => (t.time === v.time)) === i);
                chartDataRef.current = uniqueData;
                setChartData(uniqueData);
            }).catch((e) => { setDataError(`Fetch failed: ${e?.message || e}`); console.error(e); });
    }, [symbol, timeframe, cursorDate, currencyRate]);

    const loadMoreHistory = async () => {
        // Prevent simultaneous fetches or fetches when no data exists
        if (isFetchingHistory.current || !hasMoreHistory.current) return;
        
        setChartData((currentData) => {
            if (currentData.length === 0) return currentData;
            
            isFetchingHistory.current = true;
            let fetchTimeframe = timeframe;
            if (timeframe === "3M" || timeframe === "1Y" || timeframe === "ALL") fetchTimeframe = "1M";
            
            const earliestTime = currentData[0].time;
            const params = new URLSearchParams({
                symbol, interval: fetchTimeframe, limit: "1000",
                endTime: String(earliestTime * 1000 - 1),
            });
            const url = `${API_BASE}/market/klines?${params.toString()}`;

            fetch(url)
                .then(res => res.json())
                .then((data: any) => {
                    const rows: any[] = Array.isArray(data) ? data : (data?.bars || []);
                    if (rows.length === 0) {
                        hasMoreHistory.current = false;
                        isFetchingHistory.current = false;
                        return;
                    }
                    const formatted = rows.map((d: any) => ({
                        time: d[0] / 1000,
                        open: parseFloat(d[1]) * currencyRate,
                        high: parseFloat(d[2]) * currencyRate,
                        low: parseFloat(d[3]) * currencyRate,
                        close: parseFloat(d[4]) * currencyRate,
                        volume: d[5] ? parseFloat(d[5]) : 0,
                    })).filter((c: any) => Number.isFinite(c.time) && Number.isFinite(c.close) && c.close > 0);
                    
                    // Keep track of scroll offset
                    if (chartRef.current) {
                        const range = chartRef.current.timeScale().getVisibleLogicalRange();
                        if (range) prevRangeInfo.current = { ...range, dataLength: currentData.length };
                    }

                    setChartData(prev => {
                        const combined = [...formatted, ...prev];
                        const unique = combined.filter((v: any, i: number, a: any[]) => a.findIndex((t: any) => (t.time === v.time)) === i).sort((a: any, b: any) => a.time - b.time);
                        chartDataRef.current = unique;
                        return unique;
                    });
                })
                .catch(e => {
                    console.error(e);
                    isFetchingHistory.current = false;
                });
            
            return currentData;
        });
    };

        const calculateEMA = (data: any[], period: number) => {
        if (!data || data.length === 0) return [];
        const k = 2 / (period + 1);
        let emaData = [];
        let ema = data[0].close;
        for (let i = 0; i < data.length; i++) {
            ema = (data[i].close - ema) * k + ema;
            emaData.push({ time: data[i].time, value: ema });
        }
        return emaData;
    };

    const calculateSMA = (data: any[], period: number) => {
        if (!data || data.length < period) return [];
        let smaData = [];
        for (let i = period - 1; i < data.length; i++) {
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += data[i - j].close;
            }
            smaData.push({ time: data[i].time, value: sum / period });
        }
        return smaData;
    };

    const calculateVWMA = (data: any[], period: number) => {
        if (!data || data.length < period) return [];
        let vwmaData = [];
        for (let i = period - 1; i < data.length; i++) {
            let sumPriceVol = 0;
            let sumVol = 0;
            for (let j = 0; j < period; j++) {
                const vol = data[i - j].volume || 1; // Fallback to 1 if no vol
                sumPriceVol += data[i - j].close * vol;
                sumVol += vol;
            }
            vwmaData.push({ time: data[i].time, value: sumVol === 0 ? data[i].close : (sumPriceVol / sumVol) });
        }
        return vwmaData;
    };

    // Initialize Chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: '#000000' }, textColor: '#737373' },
            grid: { vertLines: { color: '#171717' }, horzLines: { color: '#171717' } },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            localization: {
                priceFormatter: price => `${currencyPrefix}${price.toFixed(2)}`
            },
            crosshair: { mode: 1, vertLine: { width: 1, color: '#404040', style: 3 }, horzLine: { width: 1, color: '#404040', style: 3 } },
            timeScale: {
                fixLeftEdge: true,
                timeVisible: true,
                secondsVisible: false,
                borderColor: '#171717',
                rightOffset: 30,
                barSpacing: 10,
                minBarSpacing: 1
            },
            rightPriceScale: { borderColor: '#171717' }
        });

                chart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
            setRenderTick(t => t + 1);
            if (logicalRange !== null) {
                // Cycle 22.4: prefetch earlier (-150 vs -50) so the next page
                // arrives BEFORE the user reaches the visible edge — eliminates
                // the visible "stop, then more loads" stutter when panning.
                if (logicalRange.from < -150) {
                    loadMoreHistory();
                }

                // Auto timeframe switching mechanism
                const visibleBars = logicalRange.to - logicalRange.from;

                const now = Date.now();
                const lastSwitch = (window as any).lastTfSwitchTime || 0;

                if (!isSwitchingTfRef.current && (now - lastSwitch > 1200)) {
                    if (visibleBars > 400 || visibleBars < 20) {
                        isSwitchingTfRef.current = true;
                        
                        setTimeframe(prevTf => {
                            const tfLevels = ["1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M", "3M", "1Y", "ALL"];
                            const currentIndex = tfLevels.indexOf(prevTf);

                            if (currentIndex !== -1) {
                                let newTf = prevTf;
                                if (visibleBars > 400 && currentIndex < tfLevels.length - 1) {
                                    newTf = tfLevels[currentIndex + 1];
                                } else if (visibleBars < 20 && currentIndex > 0) {
                                    newTf = tfLevels[currentIndex - 1];
                                }
                                
                                if (newTf !== prevTf) {
                                    if (chartRef.current && chartDataRef.current && chartDataRef.current.length > 0) {
                                        const currentRange = chartRef.current.timeScale().getVisibleLogicalRange();
                                        if (currentRange) {
                                            const fromIdx = Math.max(0, Math.floor(currentRange.from));
                                            const toIdx = Math.min(chartDataRef.current.length - 1, Math.ceil(currentRange.to));
                                            const fromTime = chartDataRef.current[fromIdx]?.time;
                                            const toTime = chartDataRef.current[toIdx]?.time;
                                            prevRangeInfo.current = { isTimeframeSwitch: true, fromTime, toTime, expand: visibleBars > 400 };
                                        }
                                    }
                                    (window as any).lastTfSwitchTime = Date.now();
                                    return newTf;
                                }
                            }
                            isSwitchingTfRef.current = false; // revert if no change
                            return prevTf;
                        });
                    }
                }
            }
        });
        chart.timeScale().subscribeVisibleTimeRangeChange(() => setRenderTick(t => t + 1));

        const handleResize = () => {
            if (chartContainerRef.current) chart.applyOptions({ width: chartContainerRef.current.clientWidth, height: chartContainerRef.current.clientHeight });
        };
        window.addEventListener('resize', handleResize);
        
        const series = chart.addCandlestickSeries({
            upColor: isVibrantColors ? '#089981' : '#34d399', 
            downColor: isVibrantColors ? '#f23645' : '#f87171', 
            borderVisible: false, 
            wickUpColor: isVibrantColors ? '#089981' : '#34d399', 
            wickDownColor: isVibrantColors ? '#f23645' : '#f87171',
        });
        
        const predictionSeries = chart.addCandlestickSeries({
            upColor: 'rgba(167, 139, 250, 0.4)',
            downColor: 'rgba(167, 139, 250, 0.4)',
            borderVisible: true,
            borderColor: 'rgba(167, 139, 250, 0.8)',
            wickUpColor: 'rgba(167, 139, 250, 0.8)',
            wickDownColor: 'rgba(167, 139, 250, 0.8)',
        });

        // Phase 1 bug fix (grid blackout): the prediction CANDLES already
        // encode the p25/p75 band via low=p25_close, high=p75_close
        // (nn_agent._render_prediction_chart). The previous implementation
        // used a stacked area-series pair where the lower (p25) layer was
        // painted opaque BLACK to mask the area beneath — but that also
        // painted over the chart's grid lines, producing the reported
        // blackout under the band. Replace both layers with thin
        // semi-transparent boundary lines so the grid stays visible.
        const predictionP75 = chart.addAreaSeries({
            topColor: 'rgba(167, 139, 250, 0.0)',
            bottomColor: 'rgba(167, 139, 250, 0.0)',
            lineColor: 'rgba(167, 139, 250, 0.45)', lineWidth: 1,
            priceLineVisible: false, lastValueVisible: false,
        });
        const predictionP25 = chart.addAreaSeries({
            topColor: 'rgba(167, 139, 250, 0.0)',      // fully transparent — grid shows through
            bottomColor: 'rgba(167, 139, 250, 0.0)',
            lineColor: 'rgba(167, 139, 250, 0.45)', lineWidth: 1,
            priceLineVisible: false, lastValueVisible: false,
        });
        // Median sits on TOP of the band — drawn last so it isn't masked.
        const predictionMedian = chart.addLineSeries({
            color: '#a78bfa', lineWidth: 2, title: 'Predicted (median)',
            priceLineVisible: false, lastValueVisible: false,
        });

        series.setData(chartData);
        predictionSeriesRef.current = predictionSeries;
        predictionMedianRef.current = predictionMedian;
        predictionP25Ref.current = predictionP25;
        predictionP75Ref.current = predictionP75;

        chartRef.current = chart;
        seriesRef.current = series;

        // Force a resize slightly after mount to ensure layout calculation.
        // G6: open on the MOST RECENT candles (scroll to real time) instead of
        // fitContent() which zooms out to show the entire history.
        const timeoutId = setTimeout(() => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
                try {
                    chart.timeScale().scrollToPosition(0, false);
                    chart.timeScale().scrollToRealTime();
                } catch { chart.timeScale().fitContent(); }
            }
        }, 100);

if (config.ema9.show) {
            const ema9 = chart.addLineSeries({ color: config.ema9.color, lineWidth: config.ema9.lineWidth as any, title: 'EMA 9' });
            ema9Ref.current = ema9;
            ema9.setData(calculateEMA(chartData, 9));
        }
        if (config.ema21.show) {
            const ema21 = chart.addLineSeries({ color: config.ema21.color, lineWidth: config.ema21.lineWidth as any, title: 'EMA 21' });
            ema21Ref.current = ema21;
            ema21.setData(calculateEMA(chartData, 21));
        }
        if (config.sma50?.show) {
            const sma50 = chart.addLineSeries({ color: config.sma50.color, lineWidth: config.sma50.lineWidth as any, title: 'SMA 50' });
            sma50Ref.current = sma50;
            sma50.setData(calculateSMA(chartData, 50));
        }
        if (config.sma200?.show) {
            const sma200 = chart.addLineSeries({ color: config.sma200.color, lineWidth: config.sma200.lineWidth as any, title: 'SMA 200' });
            sma200Ref.current = sma200;
            sma200.setData(calculateSMA(chartData, 200));
        }
        if (config.vwma?.show) {
            const vwma = chart.addLineSeries({ color: config.vwma.color, lineWidth: config.vwma.lineWidth as any, title: 'VWMA' });
            vwmaRef.current = vwma;
            vwma.setData(calculateVWMA(chartData, 20));
        }

        return () => {
            window.removeEventListener('resize', handleResize);
            clearTimeout(timeoutId);
            chart.remove();
            chartRef.current = null;
            seriesRef.current = null;
            predictionSeriesRef.current = null;
            predictionMedianRef.current = null;
            predictionP25Ref.current = null;
            predictionP75Ref.current = null;
            ema9Ref.current = null;
            ema21Ref.current = null;
            sma50Ref.current = null;
            sma200Ref.current = null;
            vwmaRef.current = null;
        };
        }, [symbol, config.ema9.show, config.ema9.color, config.ema9.lineWidth, config.ema21.show, config.ema21.color, config.ema21.lineWidth, config.sma50?.show, config.sma50?.color, config.sma50?.lineWidth, config.sma200?.show, config.sma200?.color, config.sma200?.lineWidth, config.vwma?.show, config.vwma?.color, config.vwma?.lineWidth, isVibrantColors, currencyPrefix]);

    useEffect(() => {
        if (chartData.length > 0 && seriesRef.current) {
            seriesRef.current.setData(chartData);
            
// Update EMAs and SMAs
            if (ema9Ref.current) ema9Ref.current.setData(calculateEMA(chartData, 9));
            if (ema21Ref.current) ema21Ref.current.setData(calculateEMA(chartData, 21));
            if (sma50Ref.current) sma50Ref.current.setData(calculateSMA(chartData, 50));
            if (sma200Ref.current) sma200Ref.current.setData(calculateSMA(chartData, 200));
            if (vwmaRef.current) vwmaRef.current.setData(calculateVWMA(chartData, 20));
            // Re-render EMA lines if config is ON
            // Note: In real app, you'd add ref for EMA series to replace data, but recreating chart handles it via dependency if we just let it.
            // Since we removed chartData from init dependencies, we just set main series data.

            if (prevRangeInfo.current && chartRef.current) {
                if (prevRangeInfo.current.isTimeframeSwitch && prevRangeInfo.current.fromTime !== undefined) {
                    const fromTime = prevRangeInfo.current.fromTime;
                    const toTime = prevRangeInfo.current.toTime;
                    const expand = prevRangeInfo.current.expand;

                    let fromIdx = chartData.findIndex((d: any) => d.time >= fromTime);
                    let toIdx = chartData.findIndex((d: any) => d.time >= toTime);
                    
                    if (fromIdx === -1) fromIdx = 0;
                    if (toIdx === -1) toIdx = chartData.length - 1;
                    
                    if (expand) {
                         // When zooming out over 350 bars, we switch timeframe, which means fewer new bars. 350 bars on 1m = 70 bars on 5m.
                         let mid = Math.floor((fromIdx + toIdx) / 2);
                         fromIdx = mid - 50; 
                         toIdx = mid + 50;
                    } else {
                         // When zooming in, 40 bars on 5m = 200 bars on 1m.
                         let mid = Math.floor((fromIdx + toIdx) / 2);
                         fromIdx = mid - 100;
                         toIdx = mid + 100;
                    }

                    if (fromIdx < 0) fromIdx = 0;
                    if (toIdx >= chartData.length) toIdx = chartData.length - 1;

                    chartRef.current.timeScale().setVisibleLogicalRange({
                        from: fromIdx,
                        to: toIdx
                    });
                } else if (prevRangeInfo.current.from !== undefined && !prevRangeInfo.current.isTimeframeSwitch) {
                     const newLength = chartData.length;
                     const oldLength = prevRangeInfo.current.dataLength || 0;       
                     const offset = newLength - oldLength;
                     if (offset > 0) {
                         chartRef.current.timeScale().setVisibleLogicalRange({from: prevRangeInfo.current.from + offset, to: prevRangeInfo.current.to + offset});   
                     }
                }
            } else if (chartRef.current) {
                // G6: a fresh load / symbol switch has no prior range to restore,
                // so anchor to the MOST RECENT candles instead of a random span.
                try { chartRef.current.timeScale().scrollToRealTime(); } catch { /* noop */ }
            }
            prevRangeInfo.current = null;
            isFetchingHistory.current = false;
            setTimeout(() => { isSwitchingTfRef.current = false; }, 800);
        }
    }, [chartData]);

    const handleMapToChart = (clientX: number, clientY: number) => {
        if (!chartRef.current || !seriesRef.current || !chartContainerRef.current) return null;
        const rect = chartContainerRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        const logical = chartRef.current.timeScale().coordinateToLogical(x);
        const price = seriesRef.current.coordinateToPrice(y);
        return { logical, price, x, y };
    };

    const startDrag = (e: React.MouseEvent, type: 'move'|'p1'|'p2'|'rotate', drawing: any) => {
        if (activeTool === 'eraser') {
            e.stopPropagation();
            commitDrawingUpdate(drawings.filter(x => x.id !== drawing.id));
            if (selectedDrawing === drawing.id) setSelectedDrawing(null);
            return;
        }
        if (activeTool !== 'pointer') return;
        e.stopPropagation();
        setSelectedDrawing(drawing);
        setDragCtx({ 
            id: drawing.id, type, 
            initMouseX: e.clientX, initMouseY: e.clientY, 
            startDrawing: JSON.parse(JSON.stringify(drawing)) 
        });
    };

    useEffect(() => {
        const handleWinMove = (e: MouseEvent) => {
            if (!dragCtx || !chartRef.current || !seriesRef.current || !chartContainerRef.current) return;
            const rect = chartContainerRef.current.getBoundingClientRect();
            
            setDrawings(prev => prev.map(d => {
                if (d.id !== dragCtx.id) return d;
                
                const currPath = d.path || [];
                const curLogicalX = chartRef.current.timeScale().coordinateToLogical(e.clientX - rect.left) || 0;
                const curPriceY = seriesRef.current.coordinateToPrice(e.clientY - rect.top) || 0;

                let newData = { ...d };

                if (dragCtx.type === 'move') {
                    const initMapped = handleMapToChart(dragCtx.initMouseX, dragCtx.initMouseY);
                    if (!initMapped || initMapped.logical === null) return d;

                    const deltaL = curLogicalX - initMapped.logical;
                    const deltaP = curPriceY - initMapped.price;

                    newData.l1 = (dragCtx.startDrawing.l1 || 0) + deltaL;
                    newData.p1 = (dragCtx.startDrawing.p1 || 0) + deltaP;
                    newData.l2 = (dragCtx.startDrawing.l2 || 0) + deltaL;
                    newData.p2 = (dragCtx.startDrawing.p2 || 0) + deltaP;

                    if (currPath.length > 0) {
                        newData.path = dragCtx.startDrawing.path.map((pt: any) => ({
                            l: pt.l + deltaL,
                            p: pt.p + deltaP
                        }));
                    }
                } else if (dragCtx.type === 'p1') {
                    newData.l1 = curLogicalX;
                    newData.p1 = curPriceY;
                } else if (dragCtx.type === 'p2') {
                    newData.l2 = curLogicalX;
                    newData.p2 = curPriceY;
                } else if (dragCtx.type === 'rotate' && dragCtx.startDrawing.type === 'text') {
                    // For rotate, we can compute angle
                    const c1 = chartRef.current.timeScale().logicalToCoordinate(newData.l1 as any) || 0;
                    const c2 = seriesRef.current.priceToCoordinate(newData.p1 as any) || 0;
                    const dx = e.clientX - rect.left - c1;
                    const dy = e.clientY - rect.top - c2;
                    newData.angle = Math.atan2(dy, dx) * (180 / Math.PI);
                }

                return newData;
            }));
        };

        const handleWinUp = () => setDragCtx(null);

        if (dragCtx) {
            window.addEventListener('mousemove', handleWinMove);
            window.addEventListener('mouseup', handleWinUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleWinMove);
            window.removeEventListener('mouseup', handleWinUp);
        };
    }, [dragCtx]);

    const getSnappedValue = (logicalIndex: number, price: number) => {
        if (!chartDataRef.current || chartDataRef.current.length === 0) return { logical: logicalIndex, price };
        const idx = Math.max(0, Math.min(chartDataRef.current.length - 1, Math.round(logicalIndex)));
        const candle = chartDataRef.current[idx];
        if (!candle) return { logical: idx, price };
        
        const mid = (candle.high + candle.low) / 2;
        const points = [candle.open, candle.high, candle.low, candle.close, mid];
        const closestPrice = points.reduce((prev, curr) => Math.abs(curr - price) < Math.abs(prev - price) ? curr : prev);
        
        return { logical: idx, price: closestPrice };
    };

    const onMouseDown = (e: React.MouseEvent) => {
        if (activeTool === 'pointer') { setSelectedDrawing(null); return; }
        const mapped = handleMapToChart(e.clientX, e.clientY);
        if (!mapped || mapped.logical === null) return;

        if (activeTool === 'text') {
            const txt = prompt("Enter text overlay:");
            if (txt) commitDrawingUpdate([...drawings, { id: Date.now(), type: 'text', l1: mapped.logical, p1: mapped.price, l2: mapped.logical, p2: mapped.price, txt, color: config.drawingColor }]);
            setActiveTool('pointer');
            return;
        }

        let startL1 = mapped.logical;
        let startP1 = mapped.price;

        if (activeTool === 'measure') {
            const snapped = getSnappedValue(mapped.logical, mapped.price);
            startL1 = snapped.logical;
            startP1 = snapped.price;
        }

        setIsDrawing(true);
        setCurrentDrawing({ 
            id: Date.now(), type: activeTool, 
            l1: startL1, p1: startP1, l2: startL1, p2: startP1,
            path: [{l: startL1, p: startP1}], color: config.drawingColor 
        });
    };

    const onMouseMove = (e: React.MouseEvent) => {
        if (!isDrawing || !currentDrawing) return;
        const mapped = handleMapToChart(e.clientX, e.clientY);
        if (!mapped || mapped.logical === null) return;

        if (currentDrawing.type === 'pencil' || currentDrawing.type === 'patterns') {
            setCurrentDrawing({ ...currentDrawing, path: [...currentDrawing.path, {l: mapped.logical, p: mapped.price}] });
        } else if (currentDrawing.type === 'measure') {
            const snapped = getSnappedValue(mapped.logical, mapped.price);
            setCurrentDrawing({ ...currentDrawing, l2: snapped.logical, p2: snapped.price });
        } else {
            setCurrentDrawing({ ...currentDrawing, l2: mapped.logical, p2: mapped.price });
        }
    };

    const onMouseUp = () => {
        if (isDrawing && currentDrawing) {
            commitDrawingUpdate([...drawings, currentDrawing]);
            setIsDrawing(false);
            setCurrentDrawing(null);
            if (activeTool !== 'pencil' && activeTool !== 'patterns') setActiveTool('pointer');
        }
    };

    // Keep react's fast refresh and state happy by using default values if unmapped
    const getCoordinate = (l: number, p: number) => {
        if (!chartRef.current || !seriesRef.current) return { x: -1000, y: -1000 };
        const logicalMapped = chartRef.current.timeScale().logicalToCoordinate(l as any);
        if (logicalMapped === null) return { x: -1000, y: -1000 };
        const x = logicalMapped;
        const y = seriesRef.current.priceToCoordinate(p);
        return { x: x || -1000, y: y || -1000 };
    };

    const tools = [
        { id: 'pointer', icon: <MousePointer2 size={16} /> },
        { id: 'measure', icon: <Ruler size={16} /> },
        { id: 'line', icon: <Slash size={16} /> },
        { id: 'pencil', icon: <Pencil size={16} /> },
        { id: 'text', icon: <Type size={16} /> },
        // Fib handled externally in dropdown
    ];

    return (
        <div className="w-full relative border border-[#171717] rounded-xl overflow-visible bg-[#000000] flex flex-col" style={{ minHeight: '600px' }}>
            {dataError && (
                <div className="absolute top-12 left-0 right-0 z-40 mx-3 mt-1 px-3 py-1.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[11px] font-mono pointer-events-none">
                    {dataError}
                </div>
            )}
            {/* Cycle 22.3: skeleton overlay during symbol/timeframe switch so
                the chart doesn't go visually blank for ~300 ms while
                /api/market/klines lands. */}
            {chartData.length === 0 && !dataError && (
                <div className="absolute inset-0 z-30 flex items-end gap-1 px-6 pb-16 pointer-events-none">
                    {Array.from({ length: 40 }).map((_, i) => (
                        <div
                            key={i}
                            className="chart-skeleton-bar rounded-sm flex-1"
                            style={{ height: `${15 + ((i * 17) % 55)}%`, animationDelay: `${i * 30}ms` }}
                        />
                    ))}
                </div>
            )}
            <div className="flex items-center justify-between px-3 py-2 border-b border-[#171717] bg-[#0A0A0A] relative z-50">
                <div className="flex items-center space-x-4">
                    <div className="relative" ref={symbolDropRef}>
                        <button
                            onClick={() => setShowSymbolDrop(s => !s)}
                            className="flex items-center gap-1.5 px-2 py-1 -mx-2 -my-1 rounded hover:bg-[#171717] text-[13px] font-semibold text-zinc-200 transition-colors"
                            title="Switch asset"
                        >
                            {symbol}
                            <ChevronDown size={12} className={`text-zinc-500 transition-transform ${showSymbolDrop ? "rotate-180" : ""}`} />
                        </button>
                        {showSymbolDrop && (
                            <div
                                className="absolute top-full left-0 mt-2 w-60 max-h-80 overflow-y-auto bg-[#0A0A0A] border border-[#27272a] rounded-xl shadow-2xl z-[999] p-1 transition-all duration-150"
                                style={{ animation: "chartDropdownIn 150ms ease-out both" }}
                            >
                                {/* Cycle 22.2: search filter — useful once the universe grows. */}
                                <input
                                    autoFocus
                                    type="text"
                                    placeholder="Filter symbols…"
                                    value={symbolSearch}
                                    onChange={(e) => setSymbolSearch(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === "Escape") {
                                            setShowSymbolDrop(false);
                                            setSymbolSearch("");
                                        }
                                    }}
                                    className="w-full px-3 py-1.5 mb-1 bg-[#121214] border border-[#27272a] rounded-lg text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-violet-500/50"
                                />
                                {(() => {
                                    const filter = symbolSearch.trim().toLowerCase();
                                    const match = (s: string) => !filter || s.toLowerCase().includes(filter);
                                    const cryptos = universe.crypto.filter(match);
                                    const stocks = universe.stocks.filter(match);
                                    const renderBtn = (sym: string) => (
                                        <button
                                            key={sym}
                                            onClick={() => {
                                                setShowSymbolDrop(false);
                                                setSymbolSearch("");
                                                onSymbolChange?.(sym);
                                            }}
                                            className={`w-full text-left px-3 py-1.5 rounded-lg text-xs transition-colors active:scale-[0.98] ${
                                                sym === symbol
                                                    ? "bg-zinc-800/80 text-zinc-100 font-medium"
                                                    : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
                                            }`}
                                        >
                                            {sym}
                                        </button>
                                    );
                                    return (
                                        <>
                                            {cryptos.length > 0 && (
                                                <>
                                                    <div className="px-3 pt-1 pb-1 text-[9px] uppercase tracking-widest text-zinc-500">Crypto</div>
                                                    {cryptos.map(renderBtn)}
                                                </>
                                            )}
                                            {stocks.length > 0 && (
                                                <>
                                                    <div className="px-3 pt-3 pb-1 text-[9px] uppercase tracking-widest text-zinc-500">US Stocks</div>
                                                    {stocks.map(renderBtn)}
                                                </>
                                            )}
                                            {cryptos.length === 0 && stocks.length === 0 && (
                                                <div className="px-3 py-3 text-[11px] text-zinc-500 text-center">
                                                    No symbols match &quot;{symbolSearch}&quot;
                                                </div>
                                            )}
                                        </>
                                    );
                                })()}
                                {!onSymbolChange && (
                                    <div className="px-3 py-2 text-[10px] text-amber-400">
                                        Parent didn&apos;t pass <code>onSymbolChange</code>; selection is read-only.
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="flex items-center space-x-1 border-l border-[#171717] pl-3 overflow-x-auto no-scrollbar">
                        {TIMEFRAMES.map((tf) => (
                            <button key={tf.value} onClick={() => { setTimeframe(tf.value); setCursorDate(null); }} className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${timeframe === tf.value ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}`}>
                                {tf.label}
                            </button>
                        ))}
                        <div className="relative ml-2 flex items-center">
                            <button onClick={() => setShowDatePanel(!showDatePanel)} className="p-1.5 rounded transition-colors text-zinc-500 hover:text-zinc-300 ml-1" title="Search Historical Date">
                                <Search size={14} />
                            </button>
                            <button onClick={() => {
                                if (chartRef.current && chartData.length > 0) {
                                    chartRef.current.timeScale().scrollToPosition(0, true);
                                    chartRef.current.timeScale().scrollToRealTime();
                                }
                            }} className="p-1.5 rounded transition-colors text-zinc-500 hover:text-zinc-300 ml-1" title="Go to Current Ticker">
                                <ArrowRightToLine size={14} />
                            </button>
                            <button 
                                onClick={() => setIsVibrantColors(!isVibrantColors)} 
                                className={`p-1.5 rounded transition-colors ml-1 ${isVibrantColors ? 'text-[#089981]' : 'text-zinc-500'} hover:opacity-80`} 
                                title="Toggle Vibrant Classic Theme"
                            >
                                <Palette size={14} />
                            </button>
                            {showDatePanel && (
                                <div className="absolute top-8 left-0 w-64 bg-[#0A0A0A] border border-[#171717] p-3 rounded-lg shadow-2xl z-50">
                                    <div className="text-xs text-zinc-400 mb-2">Search Historical Date</div>
                                    <div className="grid grid-cols-3 gap-2">
                                        <input type="number" placeholder="YYYY" value={optYear} onChange={e=>setOptYear(e.target.value)} className="bg-[#000000] border border-[#171717] rounded p-1 text-xs text-white outline-none" />
                                        <input type="number" placeholder="MM" value={optMonth} onChange={e=>setOptMonth(e.target.value)} className="bg-[#000000] border border-[#171717] rounded p-1 text-xs text-white outline-none" />
                                        <input type="number" placeholder="DD" value={optDay} onChange={e=>setOptDay(e.target.value)} className="bg-[#000000] border border-[#171717] rounded p-1 text-xs text-white outline-none" />
                                    </div>
                                    <button onClick={() => { const y = parseInt(optYear); const m = optMonth ? parseInt(optMonth)-1 : 0; const d = optDay ? parseInt(optDay) : 1; setCursorDate(new Date(Date.UTC(y, m, d)).getTime()); setShowDatePanel(false); }} className="w-full bg-zinc-800 hover:bg-zinc-700 text-white text-xs py-1.5 rounded mt-3">Go to Chart</button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex items-center space-x-2 relative z-50">
                    <button onClick={() => setShowPredictions(!showPredictions)} className={`flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs transition-all ${showPredictions ? 'bg-[#a78bfa]/20 text-[#a78bfa]' : 'bg-[#171717] text-zinc-300 hover:bg-[#27272a]'}`}>
                        <Brain size={14} /> <span>Predictions</span>
                    </button>
                    <div className="relative">
                        <button onClick={() => { setShowIndicators(!showIndicators); setShowSettings(false); }} className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-[#171717] text-zinc-300 text-xs hover:bg-[#27272a] transition-all">
                            <Activity size={14} /> <span>Indicators</span>
                        </button>
                        {showIndicators && (
                            <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-xl p-4 font-sans antialiased">
                                <div className="bg-[#0C0C0E] border border-neutral-800/60 rounded-2xl max-w-3xl w-full shadow-2xl overflow-hidden relative flex flex-col max-h-[85vh]">
                                    <div className="p-4 border-b border-neutral-800/60 flex justify-between items-center bg-[#121214]">
                                        <h2 className="text-xl font-semibold text-neutral-100">Indicators, Metrics & Strategies</h2>
                                        <button onClick={() => setShowIndicators(false)} className="text-neutral-500 hover:text-white transition-colors">
                                            <X size={20} />
                                        </button>
                                    </div>
                                    <div className="p-4 border-b border-neutral-800/60 bg-[#0C0C0E]">
                                        <div className="relative">
                                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-500" size={18} />
                                            <input type="text" placeholder="Search for indicators..." className="w-full bg-[#1A1A1D] border border-neutral-800/60 rounded-xl pl-10 pr-4 py-3 text-sm text-neutral-200 focus:outline-none focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20" />
                                        </div>
                                    </div>
                                    <div className="flex-1 overflow-y-auto p-2 bg-[#0C0C0E] custom-scrollbar">
                                        <div className="p-4">
                                            <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-4 px-2">Trend Indicators</div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={config.ema9?.show} onChange={(e) => setConfig({...config, ema9: {...config.ema9, show: e.target.checked}})} className="mt-1 mr-4 w-4 h-4 accent-[#38BDF8] rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">EMA 9</span><span className="text-xs text-zinc-500 mt-0.5">Exponential Moving Average 9</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={config.ema21?.show} onChange={(e) => setConfig({...config, ema21: {...config.ema21, show: e.target.checked}})} className="mt-1 mr-4 w-4 h-4 accent-[#FCD34D] rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">EMA 21</span><span className="text-xs text-zinc-500 mt-0.5">Exponential Moving Average 21</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={config.sma50?.show || false} onChange={(e) => setConfig({...config, sma50: {...config.sma50, show: e.target.checked}})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">SMA 50</span><span className="text-xs text-zinc-500 mt-0.5">Simple Moving Average 50</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={config.sma200?.show || false} onChange={(e) => setConfig({...config, sma200: {...config.sma200, show: e.target.checked}})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">SMA 200</span><span className="text-xs text-zinc-500 mt-0.5">Simple Moving Average 200</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={config.vwma?.show || false} onChange={(e) => setConfig({...config, vwma: {...config.vwma, show: e.target.checked}})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">VWMA</span><span className="text-xs text-zinc-500 mt-0.5">Volume Weighted Moving Average</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.ichimoku} onChange={(e) => setIndi({...indi, ichimoku: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Ichimoku Cloud</span><span className="text-xs text-zinc-500 mt-0.5">Ichimoku Kinko Hyo</span></div>
                                                </label>
                                            </div>
                                        </div>
                                        <div className="px-4 pb-4">
                                            <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-4 px-2">Oscillators</div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.rsi} onChange={(e) => setIndi({...indi, rsi: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">RSI</span><span className="text-xs text-zinc-500 mt-0.5">Relative Strength Index</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.macd} onChange={(e) => setIndi({...indi, macd: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">MACD</span><span className="text-xs text-zinc-500 mt-0.5">Moving Average Convergence Divergence</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.stochastic} onChange={(e) => setIndi({...indi, stochastic: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Stochastic</span><span className="text-xs text-zinc-500 mt-0.5">Stochastic Oscillator</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.cci} onChange={(e) => setIndi({...indi, cci: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">CCI</span><span className="text-xs text-zinc-500 mt-0.5">Commodity Channel Index</span></div>
                                                </label>
                                                 <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.ao} onChange={(e) => setIndi({...indi, ao: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Awesome Oscillator</span><span className="text-xs text-zinc-500 mt-0.5">AO</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.momentum} onChange={(e) => setIndi({...indi, momentum: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Momentum</span><span className="text-xs text-zinc-500 mt-0.5">Momentum Indicator</span></div>
                                                </label>
                                            </div>
                                        </div>
                                        <div className="px-4 pb-4">
                                            <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-4 px-2">Volatility & Volume</div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.bollinger} onChange={(e) => setIndi({...indi, bollinger: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Bollinger Bands</span><span className="text-xs text-zinc-500 mt-0.5">Bollinger Bands</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.atr} onChange={(e) => setIndi({...indi, atr: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">ATR</span><span className="text-xs text-zinc-500 mt-0.5">Average True Range</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.volume} onChange={(e) => setIndi({...indi, volume: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Volume</span><span className="text-xs text-zinc-500 mt-0.5">Trading Volume</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.vwap} onChange={(e) => setIndi({...indi, vwap: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">VWAP</span><span className="text-xs text-zinc-500 mt-0.5">Volume Weighted Average Price</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.obv} onChange={(e) => setIndi({...indi, obv: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">OBV</span><span className="text-xs text-zinc-500 mt-0.5">On Balance Volume</span></div>
                                                </label>
                                                 <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" checked={!!indi.cmf} onChange={(e) => setIndi({...indi, cmf: e.target.checked})} className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Chaikin Money Flow</span><span className="text-xs text-zinc-500 mt-0.5">CMF</span></div>
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                    <div className="relative">
                        <button onClick={() => { setShowSettings(!showSettings); setShowIndicators(false); }} className="p-1.5 rounded text-zinc-500 hover:text-zinc-300">
                            <Settings size={14} />
                        </button>
                        {showSettings && (
                            <div className="absolute top-10 right-0 w-72 bg-[#0A0A0A] border border-[#171717] p-3 rounded-lg shadow-2xl z-50 space-y-4">
                                <div>
                                    <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Indicator Colors</div>
                                    <div className="space-y-2">
                                        <div className="flex justify-between items-center text-xs text-zinc-300">
                                            <span>EMA 9 Color</span><input type="color" value={config.ema9.color} onChange={e=>setConfig({...config, ema9: {...config.ema9, color: e.target.value}})} className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer" />
                                        </div>
                                        <div className="flex justify-between items-center text-xs text-zinc-300">
                                            <span>EMA 21 Color</span><input type="color" value={config.ema21.color} onChange={e=>setConfig({...config, ema21: {...config.ema21, color: e.target.value}})} className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer" />
                                        </div>
                                    </div>
                                </div>
                                <div className="border-t border-[#171717]"></div>
                                <div>
                                    <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Fibonacci Settings</div>
                                    <div className="space-y-2">
                                        <div className="flex justify-between items-center text-xs text-zinc-300">
                                            <span>Line Color</span><input type="color" value={config.fibColor} onChange={e=>setConfig({...config, fibColor: e.target.value})} className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer" />
                                        </div>
                                        <div className="flex justify-between items-center text-xs text-zinc-300">
                                            <span>Brightness / Opacity</span><input type="range" min="0.1" max="1" step="0.1" value={config.fibOpacity} onChange={e=>setConfig({...config, fibOpacity: parseFloat(e.target.value)})} className="w-20 accent-zinc-500" />
                                        </div>
                                    </div>
                                </div>
                                <div className="border-t border-[#171717]"></div>
                                <div>
                                    <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">Drawing Settings</div>
                                    <div className="flex justify-between items-center text-xs text-zinc-300">
                                        <span>Default Stroke Color</span><input type="color" value={config.drawingColor} onChange={e=>setConfig({...config, drawingColor: e.target.value})} className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer" />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="flex flex-1 relative bg-[#000000]">
                <div className="flex flex-col items-center py-2 space-y-1.5 w-10 border-r border-[#171717] bg-[#0A0A0A] z-40 shrink-0">
                    {tools.map(tool => {
                        if (tool.id === 'pointer' || tool.id === 'measure' || tool.id === 'line' || tool.id === 'pencil' || tool.id === 'text') {
                            return (
                                <button
                                    key={tool.id}
                                    title={tool.id.toUpperCase()}
                                    onClick={() => setActiveTool(tool.id)}
                                    className={`p-2 rounded-md transition-all duration-100 active:scale-90 focus:outline-none focus:ring-1 focus:ring-violet-500/40 ${
                                        activeTool === tool.id
                                            ? "bg-violet-500/15 text-violet-300 ring-1 ring-violet-500/30"
                                            : "text-zinc-500 hover:bg-[#171717] hover:text-zinc-300"
                                    }`}
                                >
                                    {tool.icon}
                                </button>
                            );
                        }
                        return null;
                    })}
                    <div className="relative group">
                        <button onClick={() => setShowFibMenu(!showFibMenu)} className={`p-2 rounded-md transition-colors ${activeTool.startsWith('fib') ? 'bg-orange-500/20 text-orange-500' : 'text-zinc-500 hover:text-zinc-300 hover:bg-[#171717]'}`} title="FIBONACCI TOOLS">
                            <ListMinus size={16} />
                        </button>
                        {showFibMenu && (
                            <div className="absolute left-10 top-0 w-48 bg-[#121214] border border-[#27272a] rounded-xl shadow-2xl z-50 p-2 flex flex-col space-y-1">
                                <button onClick={() => {setActiveTool('fib'); setShowFibMenu(false);}} className={`text-left px-3 py-2 rounded-lg text-xs transition-colors ${activeTool === 'fib' ? 'bg-orange-500/20 text-orange-500' : 'text-zinc-300 hover:bg-[#27272a]'}`}>Fibonacci Retracement</button>
                                <button onClick={() => {setActiveTool('fib-extension'); setShowFibMenu(false);}} className={`text-left px-3 py-2 rounded-lg text-xs transition-colors ${activeTool === 'fib-extension' ? 'bg-orange-500/20 text-orange-500' : 'text-zinc-300 hover:bg-[#27272a]'}`}>Trend-Based Fib Extension</button>
                                <button onClick={() => {setActiveTool('fib-channel'); setShowFibMenu(false);}} className={`text-left px-3 py-2 rounded-lg text-xs transition-colors ${activeTool === 'fib-channel' ? 'bg-orange-500/20 text-orange-500' : 'text-zinc-300 hover:bg-[#27272a]'}`}>Fibonacci Channel</button>
                                <button onClick={() => {setActiveTool('fib-timezone'); setShowFibMenu(false);}} className={`text-left px-3 py-2 rounded-lg text-xs transition-colors ${activeTool === 'fib-timezone' ? 'bg-orange-500/20 text-orange-500' : 'text-zinc-300 hover:bg-[#27272a]'}`}>Fibonacci Time Zone</button>
                                <button onClick={() => {setActiveTool('fib-circles'); setShowFibMenu(false);}} className={`text-left px-3 py-2 rounded-lg text-xs transition-colors ${activeTool === 'fib-circles' ? 'bg-orange-500/20 text-orange-500' : 'text-zinc-300 hover:bg-[#27272a]'}`}>Fibonacci Circles</button>
                            </div>
                        )}
                    </div>
                    {drawings.length > 0 && (
                        <div className="flex flex-col space-y-1 mt-4 border-t border-[#171717] pt-2 w-full items-center">
                            <button onClick={() => undoDrawing()} disabled={undoStack.length === 0} className={`p-2 rounded-md transition-all ${undoStack.length > 0 ? 'text-zinc-400 hover:text-zinc-200 hover:bg-[#171717]' : 'text-zinc-700 cursor-not-allowed'}`} title="Undo">
                                <Undo size={14} />
                            </button>
                            <button onClick={() => redoDrawing()} disabled={redoStack.length === 0} className={`p-2 rounded-md transition-all ${redoStack.length > 0 ? 'text-zinc-400 hover:text-zinc-200 hover:bg-[#171717]' : 'text-zinc-700 cursor-not-allowed'}`} title="Redo">
                                <Redo size={14} />
                            </button>
                            <button onClick={() => setActiveTool('eraser')} className={`p-2 rounded-md transition-all ${activeTool === 'eraser' ? 'bg-[#171717] text-rose-400' : 'text-zinc-500 hover:text-zinc-300'}`} title="Eraser Tool">
                                <Eraser size={14} />
                            </button>
                            <button onClick={() => commitDrawingUpdate([])} className="p-2 rounded-md text-xs font-bold text-rose-600 hover:bg-[#171717] hover:text-rose-500 transition-all mt-1" title="Clear All">
                                <Trash2 size={14} />
                            </button>
                        </div>
                    )}
                </div>

                <div className="flex-1 relative flex flex-col bg-[#000000]" style={{ height: '550px' }}>
                    
                    <div ref={chartContainerRef} className="w-full flex-1 h-full" style={{ position: 'relative', zIndex: 10 }} />

                    {activeTool !== 'pointer' && (
                        <div className="absolute inset-0 z-30 cursor-crosshair" onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp} />
                    )}

                    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: selectedDrawing ? 50 : 20 }}>
                        {drawings.map(d => {
                            if (d.type === 'line') {
                                const p1 = getCoordinate(d.l1, d.p1); const p2 = getCoordinate(d.l2, d.p2);
                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={d.color || config.drawingColor} strokeWidth="5" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p2', d)} />}
                                    </g>
                                );
                            }
                            if (d.type === 'fib') {
                                const p1 = getCoordinate(d.l1, d.p1); const p2 = getCoordinate(d.l2, d.p2);
                                const diff = d.p2 - d.p1;
                                const fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={config.fibColor} strokeDasharray="4" opacity={config.fibOpacity} strokeWidth="5" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        <text x={p2.x + 10} y={p1.y} fill={config.fibColor} fontSize="10" opacity={config.fibOpacity}>0</text>
                                        <text x={p2.x + 10} y={p2.y} fill={config.fibColor} fontSize="10" opacity={config.fibOpacity}>1.0</text>
                                        {fibLevels.map(lvl => {
                                            const yLvl = seriesRef.current?.priceToCoordinate(d.p1 + (diff * lvl)) || 0;
                                            return (
                                                <g key={lvl}>
                                                    {chartContainerRef.current && <line x1={0} x2={chartContainerRef.current.clientWidth} y1={yLvl} y2={yLvl} stroke={config.fibColor} strokeWidth="1" opacity={config.fibOpacity} />}
                                                </g>
                                            )
                                        })}
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p2', d)} />}
                                    </g>
                                )
                            }
                            // Phase 5: Trend-Based Fibonacci Extension — projects fib
                            // levels (incl. >1.0) of the p1->p2 move as horizontal lines.
                            if (d.type === 'fib-extension') {
                                const p1 = getCoordinate(d.l1, d.p1); const p2 = getCoordinate(d.l2, d.p2);
                                const diff = d.p2 - d.p1;
                                const levels = [0, 0.382, 0.618, 1, 1.272, 1.618, 2, 2.618];
                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={config.fibColor} strokeDasharray="4" opacity={config.fibOpacity} strokeWidth="5" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        {levels.map(lvl => {
                                            const yLvl = seriesRef.current?.priceToCoordinate(d.p1 + (diff * lvl)) || 0;
                                            return (
                                                <g key={lvl}>
                                                    {chartContainerRef.current && <line x1={0} x2={chartContainerRef.current.clientWidth} y1={yLvl} y2={yLvl} stroke={config.fibColor} strokeWidth="1" opacity={config.fibOpacity} strokeDasharray={lvl > 1 ? "2 3" : undefined} />}
                                                    <text x={(p2.x) + 6} y={yLvl - 2} fill={config.fibColor} fontSize="9" opacity={config.fibOpacity}>{lvl.toFixed(3)}</text>
                                                </g>
                                            );
                                        })}
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p2', d)} />}
                                    </g>
                                );
                            }
                            // Phase 5: Fibonacci Channel — diagonal lines parallel (in
                            // price space) to the p1->p2 trend, spaced at fib ratios.
                            if (d.type === 'fib-channel') {
                                const diff = d.p2 - d.p1;
                                const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                                const x1 = (chartRef.current?.timeScale().logicalToCoordinate(d.l1 as any)) || 0;
                                const x2 = (chartRef.current?.timeScale().logicalToCoordinate(d.l2 as any)) || 0;
                                const pA = getCoordinate(d.l1, d.p1); const pB = getCoordinate(d.l2, d.p2);
                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        {levels.map(lvl => {
                                            const ya = seriesRef.current?.priceToCoordinate(d.p1 + diff * lvl) || 0;
                                            const yb = seriesRef.current?.priceToCoordinate(d.p2 + diff * lvl) || 0;
                                            return <line key={lvl} x1={x1} y1={ya} x2={x2} y2={yb} stroke={config.fibColor} strokeWidth={lvl === 0 || lvl === 1 ? 2 : 1} opacity={config.fibOpacity} />;
                                        })}
                                        <line x1={x1} y1={pA.y} x2={x2} y2={pB.y} stroke="transparent" strokeWidth="10" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        {selectedDrawing === d.id && <circle cx={pA.x} cy={pA.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                        {selectedDrawing === d.id && <circle cx={pB.x} cy={pB.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p2', d)} />}
                                    </g>
                                );
                            }
                            // Phase 5: Fibonacci Time Zone — vertical lines at fib-sequence
                            // bar intervals from the anchor.
                            if (d.type === 'fib-timezone') {
                                const fibBars = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55];
                                const h = chartContainerRef.current?.clientHeight || 550;
                                const pA = getCoordinate(d.l1, d.p1);
                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        {fibBars.map(fb => {
                                            const x = (chartRef.current?.timeScale().logicalToCoordinate((d.l1 + fb) as any)) || -1000;
                                            return (
                                                <g key={fb}>
                                                    <line x1={x} y1={0} x2={x} y2={h} stroke={config.fibColor} strokeWidth="1" opacity={config.fibOpacity} />
                                                    <text x={x + 3} y={12} fill={config.fibColor} fontSize="9" opacity={config.fibOpacity}>{fb}</text>
                                                </g>
                                            );
                                        })}
                                        <line x1={pA.x} y1={0} x2={pA.x} y2={h} stroke="transparent" strokeWidth="10" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        {selectedDrawing === d.id && <circle cx={pA.x} cy={pA.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                    </g>
                                );
                            }
                            // Phase 5: Fibonacci Circles — concentric ellipses centred at
                            // the anchor, radii at fib ratios of the p1->p2 distance.
                            if (d.type === 'fib-circles') {
                                const p1 = getCoordinate(d.l1, d.p1); const p2 = getCoordinate(d.l2, d.p2);
                                const rx = Math.abs(p2.x - p1.x); const ry = Math.abs(p2.y - p1.y);
                                const levels = [0.236, 0.382, 0.5, 0.618, 1];
                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        {levels.map(lvl => (
                                            <ellipse key={lvl} cx={p1.x} cy={p1.y} rx={rx * lvl} ry={ry * lvl} fill="none" stroke={config.fibColor} strokeWidth="1" opacity={config.fibOpacity} />
                                        ))}
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={config.fibColor} strokeDasharray="4" opacity={config.fibOpacity} strokeWidth="5" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p2', d)} />}
                                    </g>
                                );
                            }
                            if (d.type === 'pencil' || d.type === 'patterns') {
                                const pts = d.path.map((p:any) => getCoordinate(p.l, p.p));
                                const dPath = pts.map((p:any, i:number) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');        
                                return <path key={d.id} d={dPath} fill="transparent" stroke={d.color || config.drawingColor} strokeWidth="5" style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }} cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />;
                            }
                            if (d.type === 'measure') {
                                const p1 = getCoordinate(d.l1, d.p1); 
                                const p2 = getCoordinate(d.l2, d.p2);
                                const priceDiff = d.p2 - d.p1;
                                const pricePerc = d.p1 !== 0 ? (priceDiff / d.p1) * 100 : 0;
                                const barsDiff = Math.round(d.l2 - d.l1);
                                
                                const isUp = d.p2 >= d.p1;
                                const boxColor = isUp ? 'rgba(52, 211, 153, 0.2)' : 'rgba(248, 113, 113, 0.2)';
                                const strokeColor = isUp ? '#34d399' : '#f87171';
                                
                                const xMin = Math.min(p1.x, p2.x);
                                const xMax = Math.max(p1.x, p2.x);
                                const yMin = Math.min(p1.y, p2.y);
                                const yMax = Math.max(p1.y, p2.y);

                                const isExpanded = expandedMeasure === d.id;
                                const t1 = chartDataRef.current[Math.round(d.l1)]?.time as number;
                                const t2 = chartDataRef.current[Math.round(d.l2)]?.time as number;
                                let timeStr = "";
                                if (t1 && t2) {
                                    const ms = Math.abs(t2 - t1) * 1000;
                                    const days = Math.floor(ms / (1000 * 60 * 60 * 24));
                                    const hours = Math.floor((ms / (1000 * 60 * 60)) % 24);
                                    const mins = Math.floor((ms / (1000 * 60)) % 60);
                                    if (days > 0) timeStr += `${days}d `;
                                    if (hours > 0) timeStr += `${hours}h `;
                                    if (mins > 0 || timeStr === "") timeStr += `${mins}m`;
                                } else {
                                    timeStr = "Unknown";
                                }
                                
                                const boxWidth = isExpanded ? 220 : 160;
                                const boxHeight = isExpanded ? 80 : 40;

                                return (
                                    <g key={d.id} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        <rect x={xMin} y={yMin} width={Math.max(0, xMax - xMin)} height={Math.max(0, yMax - yMin)} fill={boxColor} stroke="none" />
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p1.y} stroke={strokeColor} strokeWidth="1" strokeDasharray="4" />
                                        <line x1={p2.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={strokeColor} strokeWidth="1" strokeDasharray="4" />
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={strokeColor} strokeWidth="2" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} />
                                        
                                        <rect x={p2.x + 10} y={p2.y - 15} width={boxWidth} height={boxHeight} fill="#000000" stroke={strokeColor} rx="4" />
                                        <text x={p2.x + 15} y={p2.y} fill={strokeColor} fontSize="12" fontFamily="monospace">
                                            {priceDiff > 0 ? '+' : ''}{priceDiff.toFixed(2)} ({pricePerc > 0 ? '+' : ''}{pricePerc.toFixed(2)}%)
                                        </text>
                                        <text x={p2.x + 15} y={p2.y + 15} fill="#a1a1aa" fontSize="11" fontFamily="monospace">
                                            {Math.abs(barsDiff)} Bars ({timeStr})
                                        </text>
                                        
                                        {/* Toggle button */}
                                        <g cursor="pointer" style={{pointerEvents: 'auto'}} onMouseDown={(e) => { e.stopPropagation(); setExpandedMeasure(isExpanded ? null : d.id); }}>
                                            <rect x={p2.x + 10 + boxWidth - 25} y={p2.y - 10} width="20" height="20" fill="#171717" stroke={strokeColor} rx="2" />
                                            <text x={p2.x + 10 + boxWidth - 19} y={p2.y + 4} fill="#a1a1aa" fontSize="14" fontWeight="bold" style={{userSelect:'none'}}>
                                                {isExpanded ? '-' : '+'}
                                            </text>
                                        </g>
                                        
                                        {isExpanded && (
                                            <>
                                            <text x={p2.x + 15} y={p2.y + 35} fill="#71717a" fontSize="10" fontFamily="monospace">
                                                Start: {t1 ? new Date(t1 * 1000).toLocaleString() : 'N/A'}
                                            </text>
                                            <text x={p2.x + 15} y={p2.y + 50} fill="#71717a" fontSize="10" fontFamily="monospace">
                                                End:   {t2 ? new Date(t2 * 1000).toLocaleString() : 'N/A'}
                                            </text>
                                            </>
                                        )}

                                        
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p1', d)} />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" cursor="grab" onMouseDown={(e) => startDrag(e, 'p2', d)} />}
                                    </g>
                                );
                            }
                            if (d.type === 'text') {
                                const p = getCoordinate(d.l1, d.p1);
                                const rotation = d.angle || 0;
                                return (
                                    <g key={d.id} transform={`rotate(${rotation} ${p.x} ${p.y})`} style={{ pointerEvents: (activeTool === 'pointer' || activeTool === 'eraser') ? 'auto' : 'none' }}>
                                        <text x={p.x} y={p.y} fill={d.color || config.drawingColor} fontSize="16" cursor="move" onMouseDown={(e)=>{startDrag(e, 'move', d);}} fontFamily="monospace" style={{userSelect: 'none'}}>{d.txt}</text>
                                        {selectedDrawing === d.id && <circle cx={p.x + 20} cy={p.y - 20} r="6" fill="#ffffff" cursor="alias" onMouseDown={(e) => startDrag(e, 'rotate', d)} />}
                                    </g>
                                );
                            }
                            return null;
                        })}

                        {isDrawing && currentDrawing && currentDrawing.type === 'line' && (
                            <line x1={getCoordinate(currentDrawing.l1, currentDrawing.p1).x} y1={getCoordinate(currentDrawing.l1, currentDrawing.p1).y} x2={getCoordinate(currentDrawing.l2, currentDrawing.p2).x} y2={getCoordinate(currentDrawing.l2, currentDrawing.p2).y} stroke={config.drawingColor} strokeWidth="2" />
                        )}
                        {isDrawing && currentDrawing && currentDrawing.type === 'measure' && (() => {
                            const p1 = getCoordinate(currentDrawing.l1, currentDrawing.p1); 
                            const p2 = getCoordinate(currentDrawing.l2, currentDrawing.p2);
                            const priceDiff = currentDrawing.p2 - currentDrawing.p1;
                            const pricePerc = currentDrawing.p1 !== 0 ? (priceDiff / currentDrawing.p1) * 100 : 0;
                            const barsDiff = Math.round(currentDrawing.l2 - currentDrawing.l1);
                            
                            const isUp = currentDrawing.p2 >= currentDrawing.p1;
                            const boxColor = isUp ? 'rgba(52, 211, 153, 0.2)' : 'rgba(248, 113, 113, 0.2)';
                            const strokeColor = isUp ? '#34d399' : '#f87171';
                            
                            const xMin = Math.min(p1.x, p2.x);
                            const xMax = Math.max(p1.x, p2.x);
                            const yMin = Math.min(p1.y, p2.y);
                            const yMax = Math.max(p1.y, p2.y);

                            const t1 = chartDataRef.current[Math.round(currentDrawing.l1)]?.time as number;
                            const t2 = chartDataRef.current[Math.round(currentDrawing.l2)]?.time as number;
                            let timeStr = "";
                            if (t1 && t2) {
                                const ms = Math.abs(t2 - t1) * 1000;
                                const days = Math.floor(ms / (1000 * 60 * 60 * 24));
                                const hours = Math.floor((ms / (1000 * 60 * 60)) % 24);
                                const mins = Math.floor((ms / (1000 * 60)) % 60);
                                if (days > 0) timeStr += `${days}d `;
                                if (hours > 0) timeStr += `${hours}h `;
                                if (mins > 0 || timeStr === "") timeStr += `${mins}m`;
                            } else {
                                timeStr = "Unknown";
                            }

                            return (
                                <g style={{ pointerEvents: 'none' }}>
                                    <rect x={xMin} y={yMin} width={Math.max(0, xMax - xMin)} height={Math.max(0, yMax - yMin)} fill={boxColor} stroke="none" />
                                    <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p1.y} stroke={strokeColor} strokeWidth="1" strokeDasharray="4" />
                                    <line x1={p2.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={strokeColor} strokeWidth="1" strokeDasharray="4" />
                                    <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={strokeColor} strokeWidth="2" />
                                    
                                    <rect x={p2.x + 10} y={p2.y - 15} width="160" height="40" fill="#000000" stroke={strokeColor} rx="4" />
                                    <text x={p2.x + 15} y={p2.y} fill={strokeColor} fontSize="12" fontFamily="monospace">
                                        {priceDiff > 0 ? '+' : ''}{priceDiff.toFixed(2)} ({pricePerc > 0 ? '+' : ''}{pricePerc.toFixed(2)}%)
                                    </text>
                                    <text x={p2.x + 15} y={p2.y + 15} fill="#a1a1aa" fontSize="11" fontFamily="monospace">
                                        {Math.abs(barsDiff)} Bars ({timeStr})
                                    </text>
                                </g>
                            );
                        })()}
                        {isDrawing && currentDrawing && currentDrawing.type === 'fib' && (
                            <line x1={getCoordinate(currentDrawing.l1, currentDrawing.p1).x} y1={getCoordinate(currentDrawing.l1, currentDrawing.p1).y} x2={getCoordinate(currentDrawing.l2, currentDrawing.p2).x} y2={getCoordinate(currentDrawing.l2, currentDrawing.p2).y} stroke={config.fibColor} strokeWidth="2" strokeDasharray="4" />
                        )}
                        {isDrawing && currentDrawing && (currentDrawing.type === 'pencil' || currentDrawing.type === 'patterns') && (
                            <path d={currentDrawing.path.map((pt:any) => getCoordinate(pt.l, pt.p)).map((px:any, i:any) => (i === 0 ? `M ${px.x} ${px.y}` : `L ${px.x} ${px.y}`)).join(' ')} fill="transparent" stroke={config.drawingColor} strokeWidth="2" />
                        )}
                    </svg>

                    {selectedDrawing && (
                        <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center space-x-2 bg-[#0A0A0A] border border-[#171717] p-1.5 rounded-lg z-50 shadow-xl">
                            <span className="text-[10px] text-zinc-500 px-2">EDIT DRAWING</span>
                            <div className="h-3 w-px bg-[#171717] mx-1"></div>
                            <input type="color" title="Change Color" value={drawings.find(d=>d.id===selectedDrawing)?.color || config.drawingColor} onChange={(e) => commitDrawingUpdate(drawings.map(d => d.id === selectedDrawing ? {...d, color: e.target.value} : d))} className="w-5 h-5 rounded cursor-pointer" style={{background: 'transparent', border: 0, padding: 0}} />
                            <div className="h-3 w-px bg-[#171717] mx-1"></div>
                            <button onClick={() => { commitDrawingUpdate(drawings.filter(d => d.id !== selectedDrawing)); setSelectedDrawing(null); }} className="p-1 hover:bg-[#171717] rounded text-rose-500" title="Delete"><Trash2 size={16}/></button>
                            <button onClick={() => setSelectedDrawing(null)} className="p-1 hover:bg-[#171717] rounded text-zinc-400"><X size={16}/></button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
