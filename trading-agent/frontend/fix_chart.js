const fs = require('fs');

const chartCode = `
"use client";

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi } from "lightweight-charts";
import { Search, Pencil, Type, Activity, TrendingUp, MousePointer2, Slash } from "lucide-react";

const TIMEFRAMES = [
    { label: "1m", value: "1m" },
    { label: "5m", value: "5m" },
    { label: "15m", value: "15m" },
    { label: "1h", value: "1h" },
    { label: "4h", value: "4h" },
    { label: "1D", value: "1d" },
    { label: "1W", value: "1w" }
];

export default function TradingChart({ symbol = "BTCUSDT" }: { symbol?: string }) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const [chartData, setChartData] = useState<any[]>([]);
    const [timeframe, setTimeframe] = useState("15m");
    const [cursorDate, setCursorDate] = useState<number | null>(null);
    
    // Date Pannel State
    const [showDatePanel, setShowDatePanel] = useState(false);
    const [optYear, setOptYear] = useState("");
    const [optMonth, setOptMonth] = useState("");
    const [optDay, setOptDay] = useState("");
    const [activeTool, setActiveTool] = useState("pointer");

    useEffect(() => {
        let url = \`https://api.binance.com/api/v3/klines?symbol=\${symbol}&interval=\${timeframe}&limit=1000\`;
        if (cursorDate) {
            url += \`&startTime=\${cursorDate}\`;
        }

        fetch(url)
            .then(res => res.json())
            .then(data => {
                const formatted = data.map((d: any) => ({
                    time: d[0] / 1000,
                    open: parseFloat(d[1]),
                    high: parseFloat(d[2]),
                    low: parseFloat(d[3]),
                    close: parseFloat(d[4]),
                }));
                // Binance returns data chronologically, lightweight-charts expects it historically ordered
                setChartData(formatted);
            })
            .catch(err => console.error(err));
    }, [symbol, timeframe, cursorDate]);

    useEffect(() => {
        if (!chartContainerRef.current || chartData.length === 0) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { color: '#000000' },
                textColor: '#737373',
            },
            grid: {
                vertLines: { color: '#171717' },
                horzLines: { color: '#171717' },
            },
            width: chartContainerRef.current.clientWidth,
            height: 500,
            crosshair: {
                mode: 1,
                vertLine: { width: 1, color: '#404040', style: 3 },
                horzLine: { width: 1, color: '#404040', style: 3 },
            },
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: '#171717',
                rightOffset: 12,
            },
            rightPriceScale: {
                borderColor: '#171717',
            }
        });

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
            }
        };

        window.addEventListener('resize', handleResize);
        chartRef.current = chart;

        // Vibrant but organic colors
        const candlestickSeries = chart.addCandlestickSeries({
            upColor: '#34d399', // Vibrant Green
            downColor: '#f87171', // Vibrant Terracotta Soft
            borderVisible: false,
            wickUpColor: '#34d399',
            wickDownColor: '#f87171',
        });

        candlestickSeries.setData(chartData);

        if (chartData.length > 0) {
            chart.timeScale().fitContent();
        }

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

        const ema9 = chart.addLineSeries({ color: '#38BDF8', lineWidth: 1, title: 'EMA 9' });
        ema9.setData(calculateEMA(chartData, 9));

        const ema21 = chart.addLineSeries({ color: '#FCD34D', lineWidth: 2, title: 'EMA 21' });
        ema21.setData(calculateEMA(chartData, 21));

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [chartData]);

    const handleSearchDate = () => {
        if (!optYear) return;
        const y = parseInt(optYear);
        const m = optMonth ? parseInt(optMonth) - 1 : 0;
        const d = optDay ? parseInt(optDay) : 1;
        
        const targetDate = new Date(Date.UTC(y, m, d));
        setCursorDate(targetDate.getTime());
        
        if (!optMonth) setTimeframe("1d");
        else if (!optDay) setTimeframe("1w");
        else setTimeframe("4h");

        setShowDatePanel(false);
    };

    const tools = [
        { id: 'pointer', icon: <MousePointer2 size={16} /> },
        { id: 'line', icon: <Slash size={16} /> },
        { id: 'pencil', icon: <Pencil size={16} /> },
        { id: 'text', icon: <Type size={16} /> },
        { id: 'indicators', icon: <Activity size={16} /> },
        { id: 'patterns', icon: <TrendingUp size={16} /> },
    ];

    return (
        <div className="w-full relative border border-[#171717] rounded-xl overflow-hidden bg-[#000000] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#171717] bg-[#0A0A0A] relative z-20">
                <div className="flex items-center space-x-6">
                    <div className="text-[14px] font-semibold text-zinc-200 tracking-wide">{symbol}</div>
                    
                    <div className="flex items-center space-x-1 border-l border-[#171717] pl-4">
                        {TIMEFRAMES.map((tf) => (
                            <button
                                key={tf.value}
                                onClick={() => { setTimeframe(tf.value); setCursorDate(null); }}
                                className={\`px-2 py-1 rounded text-xs transition-colors \${timeframe === tf.value ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}\`}
                            >
                                {tf.label}
                            </button>
                        ))}
                        
                        <div className="relative ml-2">
                            <button 
                                onClick={() => setShowDatePanel(!showDatePanel)}
                                className={\`p-1.5 rounded transition-colors ml-2 flex items-center justify-center \${showDatePanel ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}\`}
                            >
                                <Search size={14} />
                            </button>
                            
                            {showDatePanel && (
                                <div className="absolute top-8 left-0 w-64 bg-[#0A0A0A] border border-[#171717] p-3 rounded-lg shadow-2xl flex flex-col space-y-3 z-50">
                                    <div className="text-xs text-zinc-400 font-medium">Jump to Date</div>
                                    <div className="grid grid-cols-3 gap-2">
                                        <input type="number" placeholder="YYYY" value={optYear} onChange={e=>setOptYear(e.target.value)} className="bg-[#000000] border border-[#171717] rounded p-1.5 text-xs text-white placeholder-zinc-700 outline-none" />
                                        <input type="number" placeholder="MM" value={optMonth} onChange={e=>setOptMonth(e.target.value)} className="bg-[#000000] border border-[#171717] rounded p-1.5 text-xs text-white placeholder-zinc-700 outline-none" min="1" max="12" />
                                        <input type="number" placeholder="DD" value={optDay} onChange={e=>setOptDay(e.target.value)} className="bg-[#000000] border border-[#171717] rounded p-1.5 text-xs text-white placeholder-zinc-700 outline-none" min="1" max="31" />
                                    </div>
                                    <button onClick={handleSearchDate} className="bg-zinc-800 hover:bg-zinc-700 text-white text-xs py-1.5 rounded transition-all mt-1">Go to Chart</button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex items-center space-x-3 text-xs text-zinc-500 pr-4 hidden md:flex">
                    <div className="flex items-center space-x-1"><div className="w-2 h-2 rounded-full bg-[#38BDF8]"></div><span>EMA 9</span></div>
                    <div className="flex items-center space-x-1"><div className="w-2 h-2 rounded-full bg-[#FCD34D]"></div><span>EMA 21</span></div>
                </div>
            </div>

            <div className="flex flex-1 relative" style={{ minHeight: '450px'}}>
                <div className="flex flex-col items-center py-2 space-y-1.5 w-10 border-r border-[#171717] bg-[#0A0A0A] z-10 shrink-0">
                    {tools.map(tool => (
                        <button
                            key={tool.id}
                            onClick={() => {
                                setActiveTool(tool.id);
                                if(tool.id === "patterns" || tool.id === "pencil") alert("Advanced drawing tools require TradingView license.");
                            }}
                            className={\`p-2 rounded-md transition-colors \${activeTool === tool.id ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}\`}
                        >
                            {tool.icon}
                        </button>
                    ))}
                </div>

                <div ref={chartContainerRef} className="w-full flex-1 touch-none" />
            </div>
        </div>
    );
}

`;

fs.writeFileSync('./components/TradingChart.tsx', chartCode);

let navCode = fs.readFileSync('./components/Navbar.tsx', 'utf8');
navCode = navCode.replace(/#A6622E/g, '#C45A3E');
fs.writeFileSync('./components/Navbar.tsx', navCode);

console.log('Update Complete.');
