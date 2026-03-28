import { useState, useEffect } from "react";
import { subscribeToLiveWs, fetchFromAPI } from "../api";

export function useMarketData(symbol: string) {
    const [klines, setKlines] = useState<any[]>([]);
    const [orderbook, setOrderbook] = useState({ bids: [], asks: [] });
    const [tradeHistory, setTradeHistory] = useState<any[]>([]);

    useEffect(() => {
        // Initial fetch
        fetchFromAPI(`/market/klines?symbol=${symbol}&interval=1m&limit=100`)
            .then(data => data && setKlines(data))
            .catch(console.error);

        fetchFromAPI(`/market/depth?symbol=${symbol}`)
            .then(data => data && setOrderbook(data))
            .catch(console.error);

        fetchFromAPI(`/market/trades?symbol=${symbol}`)
            .then(data => data && setTradeHistory(data))
            .catch(console.error);

        // Subscribe to live websocket updates
        const ws = subscribeToLiveWs((topic, data) => {
            if (topic === "kline" && data.symbol === symbol) {
                setKlines(prev => {
                    const next = [...prev];
                    const last = next[next.length - 1];
                    if (last && last.time === data.time) {
                        next[next.length - 1] = data;
                    } else {
                        next.push(data);
                        if (next.length > 100) next.shift();
                    }
                    return next;
                });
            } else if (topic === "depth" && data.symbol === symbol) {
                setOrderbook(data);
            } else if (topic === "trade" && data.symbol === symbol) {
                setTradeHistory(prev => [data, ...prev].slice(0, 50));
            }
        });

        return () => ws.close();
    }, [symbol]);

    return { klines, orderbook, tradeHistory };
}
