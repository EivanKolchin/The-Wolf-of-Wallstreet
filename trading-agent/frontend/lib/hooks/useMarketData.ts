import { useState, useEffect } from "react";
import { subscribeToLiveWs, fetchFromAPI } from "../api";

export function useMarketData(symbol: string) {
    const [klines, setKlines] = useState<any[]>([]);
    const [orderbook, setOrderbook] = useState({ bids: [], asks: [] });
    const [tradeHistory, setTradeHistory] = useState<any[]>([]);

    useEffect(() => {
        // Initial fetch
        fetchFromAPI(`/market/klines?symbol=${symbol}&interval=1m&limit=100`)
            .then(data => {
                if (data) {
                    const formatted = data.map((d: any) => ({
                        time: d[0],
                        open: parseFloat(d[1]),
                        high: parseFloat(d[2]),
                        low: parseFloat(d[3]),
                        close: parseFloat(d[4]),
                        volume: parseFloat(d[5]),
                    }));
                    setKlines(formatted);
                }
            })
            .catch(console.error);

        fetchFromAPI(`/market/depth?symbol=${symbol}`)
            .then(data => data && setOrderbook(data))
            .catch(console.error);

        fetchFromAPI(`/market/trades?symbol=${symbol}`)
            .then(data => data && setTradeHistory(data))
            .catch(console.error);

        // Subscribe to live websocket updates from the backend (for predictions, signals, etc)
        const backendWs = subscribeToLiveWs((topic, data) => {
            // Unlikely to get market updates here anymore but keep for compatibility
        });

        // Set up direct connection to Binance WS to sync prices exactly with TradingView
        const safeSymbol = symbol.toLowerCase();
        const binanceWs = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${safeSymbol}@kline_1m`);
        
        binanceWs.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload && payload.data && payload.data.e === 'kline') {
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
