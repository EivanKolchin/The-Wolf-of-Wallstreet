"use client";

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi } from "lightweight-charts";

export default function TradingChart({ symbol = "BTCUSDT" }: { symbol?: string }) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const [chartData, setChartData] = useState<any[]>([]);

    useEffect(() => {
        // Fetch historical data from Binance Public API
        fetch(`https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=5m&limit=200`)
            .then(res => res.json())
            .then(data => {
                const formatted = data.map((d: any) => ({
                    time: d[0] / 1000,
                    open: parseFloat(d[1]),
                    high: parseFloat(d[2]),
                    low: parseFloat(d[3]),
                    close: parseFloat(d[4]),
                }));
                setChartData(formatted);
            })
            .catch(err => console.error(err));
    }, [symbol]);

    useEffect(() => {
        if (!chartContainerRef.current || chartData.length === 0) return;

        const chart: IChartApi = createChart(chartContainerRef.current, {
            layout: {
                background: { color: 'transparent' },
                textColor: '#A1A1AA',
            },
            grid: {
                vertLines: { color: '#27272A' },
                horzLines: { color: '#27272A' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 400,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
            }
        });

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);

        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#22C55E',
            downColor: '#EF4444',
            borderVisible: false,
            wickUpColor: '#22C55E',
            wickDownColor: '#EF4444'
        });
        
        candlestickSeries.setData(chartData);

        // Simple EMA calculations
        const calculateEMA = (data: any[], period: number) => {
            const k = 2 / (period + 1);
            let emaData = [];
            let ema = data[0].close;
            for (let i = 0; i < data.length; i++) {
                ema = (data[i].close - ema) * k + ema;
                emaData.push({ time: data[i].time, value: ema });
            }
            return emaData;
        };

        const ema9 = chart.addLineSeries({ color: '#60A5FA', lineWidth: 1 });
        ema9.setData(calculateEMA(chartData, 9));

        const ema21 = chart.addLineSeries({ color: '#FCD34D', lineWidth: 2 });
        ema21.setData(calculateEMA(chartData, 21));

        const ema200 = chart.addLineSeries({ color: '#A78BFA', lineWidth: 3 });
        ema200.setData(calculateEMA(chartData, 200));

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [chartData]);

    return (
        <div className="w-full relative h-[400px] border border-zinc-800 rounded-xl overflow-hidden bg-zinc-900/50">
            <div className="absolute top-4 left-4 z-10 text-white font-semibold flex items-center space-x-2 bg-zinc-950/80 px-3 py-1 rounded-md border border-zinc-800">
                <span>{symbol}</span>
                <span className="text-xs text-zinc-400 font-normal">5m Candlesticks</span>
            </div>
            <div ref={chartContainerRef} className="w-full h-full" />
        </div>
    );
}