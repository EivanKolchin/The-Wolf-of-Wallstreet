import { useState, useEffect } from "react";
import { subscribeToLiveWs, fetchFromAPI, WS_BASE } from "../api";

const STOCK_TICKERS = new Set([
    "SNDK", "AMD", "MU", "BE", "NVDA", "TSM", "SMCI", "TSLA", "MSTR", "COIN", "PLTR",
]);

function isStockSymbol(symbol: string) {
    const s = (symbol || "").toUpperCase();
    return STOCK_TICKERS.has(s) || (!s.endsWith("USDT") && s.length <= 5);
}

export function useMarketData(symbol: string) {
    const [klines, setKlines] = useState<any[]>([]);
    const [orderbook, setOrderbook] = useState({ bids: [], asks: [] });
    const [tradeHistory, setTradeHistory] = useState<any[]>([]);

    useEffect(() => {
        const stock = isStockSymbol(symbol);

        const formatKlines = (data: any) => {
            const rows = Array.isArray(data) ? data : (data?.bars || []);
            return rows.map((d: any) => ({
                time: d[0],
                open: parseFloat(d[1]),
                high: parseFloat(d[2]),
                low: parseFloat(d[3]),
                close: parseFloat(d[4]),
                volume: parseFloat(d[5]),
            }));
        };

        fetchFromAPI(`/market/klines?symbol=${symbol}&interval=1m&limit=100`)
            .then(data => {
                if (data) setKlines(formatKlines(data));
            })
            .catch(console.error);

        fetchFromAPI(`/market/depth?symbol=${symbol}`)
            .then(data => data && setOrderbook(data))
            .catch(console.error);

        fetchFromAPI(`/market/trades?symbol=${symbol}`)
            .then(data => data && setTradeHistory(data))
            .catch(console.error);

        const backendWs = subscribeToLiveWs(() => {});

        if (stock) {
            const wsUrl = `${WS_BASE}/stocks?symbol=${encodeURIComponent(symbol)}`;
            const ws = new WebSocket(wsUrl);
            ws.onmessage = (event) => {
                try {
                    const m = JSON.parse(event.data);
                    const price = m?.type === "trade" ? m.price : m?.type === "quote" ? m.mid : null;
                    if (typeof price !== "number" || price <= 0) return;
                    setKlines(prev => {
                        if (prev.length === 0) return prev;
                        const next = [...prev];
                        const last = next[next.length - 1];
                        const updated = {
                            ...last,
                            high: Math.max(last.high, price),
                            low: Math.min(last.low, price),
                            close: price,
                        };
                        next[next.length - 1] = updated;
                        return next;
                    });
                } catch (e) {
                    console.error("Stock WS error", e);
                }
            };
            const refresh = setInterval(() => {
                fetchFromAPI(`/market/klines?symbol=${symbol}&interval=1m&limit=100`)
                    .then(data => { if (data) setKlines(formatKlines(data)); })
                    .catch(() => {});
            }, 60_000);
            return () => {
                backendWs.close();
                ws.close();
                clearInterval(refresh);
            };
        }

        const safeSymbol = symbol.toLowerCase();
        const binanceWs = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${safeSymbol}@kline_1m`);
        binanceWs.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload?.data?.e === 'kline') {
                    const k = payload.data.k;
                    const newCandle = {
                        time: k.t,
                        open: parseFloat(k.o),
                        high: parseFloat(k.h),
                        low: parseFloat(k.l),
                        close: parseFloat(k.c),
                        volume: parseFloat(k.v),
                    };
                    setKlines(prev => {
                        const next = [...prev];
                        if (next.length === 0) return [newCandle];
                        const last = next[next.length - 1];
                        if (last && last.time === newCandle.time) {
                            next[next.length - 1] = newCandle;
                        } else if (newCandle.time > last.time) {
                            next.push(newCandle);
                            if (next.length > 100) next.shift();
                        }
                        return next;
                    });
                }
            } catch (e) {
                console.error("Binance WS error", e);
            }
        };

        return () => {
            backendWs.close();
            binanceWs.close();
        };
    }, [symbol]);

    return { klines, orderbook, tradeHistory };
}
