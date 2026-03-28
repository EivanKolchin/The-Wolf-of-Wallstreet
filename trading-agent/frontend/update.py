import os

code = r"""
"use client";

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi } from "lightweight-charts";
import { Search, Pencil, Type, Activity, MousePointer2, Slash, Settings, Trash2, ListMinus, X } from "lucide-react";

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
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

    const [chartData, setChartData] = useState<any[]>([]);
    const [timeframe, setTimeframe] = useState("15m");
    const [cursorDate, setCursorDate] = useState<number | null>(null);
    
    // UI Panels
    const [showDatePanel, setShowDatePanel] = useState(false);
    const [showIndicators, setShowIndicators] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    
    // Date states
    const [optYear, setOptYear] = useState("");
    const [optMonth, setOptMonth] = useState("");
    const [optDay, setOptDay] = useState("");

    // Customization Settings
    const [config, setConfig] = useState({
        ema9: { show: true, color: '#38BDF8', lineWidth: 1 },
        ema21: { show: true, color: '#FCD34D', lineWidth: 2 },
        fibColor: '#A78BFA',
        fibOpacity: 0.6,
        drawingColor: '#34d399'
    });

    // Drawing Engine States
    const [activeTool, setActiveTool] = useState("pointer");
    const [drawings, setDrawings] = useState<any[]>([]);
    const [isDrawing, setIsDrawing] = useState(false);
    const [currentDrawing, setCurrentDrawing] = useState<any>(null);
    const [selectedDrawing, setSelectedDrawing] = useState<number | null>(null);
    const [renderTick, setRenderTick] = useState(0);

    // Fetch Data
    useEffect(() => {
        let url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${timeframe}&limit=1000`;
        if (cursorDate) url += `&startTime=${cursorDate}`;

        fetch(url)
            .then(res => res.json())
            .then(data => {
                const formatted = data.map((d: any) => ({
                    time: d[0] / 1000, open: parseFloat(d[1]), high: parseFloat(d[2]), low: parseFloat(d[3]), close: parseFloat(d[4]),
                }));
                const uniqueData = formatted.filter((v: any, i: number, a: any[]) => a.findIndex((t: any) => (t.time === v.time)) === i);
                setChartData(uniqueData);
            }).catch(console.error);
    }, [symbol, timeframe, cursorDate]);

    // Initialize Chart
    useEffect(() => {
        if (!chartContainerRef.current || chartData.length === 0) return;

        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: 'solid', color: '#000000' }, textColor: '#737373' },
            grid: { vertLines: { color: '#171717' }, horzLines: { color: '#171717' } },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
            crosshair: { mode: 1, vertLine: { width: 1, color: '#404040', style: 3 }, horzLine: { width: 1, color: '#404040', style: 3 } },
            timeScale: { 
                timeVisible: true, 
                secondsVisible: false, 
                borderColor: '#171717', 
                rightOffset: 12,
                barSpacing: 10,
                minBarSpacing: 1
            },
            rightPriceScale: { borderColor: '#171717' }
        });

        chart.timeScale().subscribeVisibleLogicalRangeChange(() => setRenderTick(t => t + 1));
        chart.timeScale().subscribeVisibleTimeRangeChange(() => setRenderTick(t => t + 1));

        const handleResize = () => {
            if (chartContainerRef.current) chart.applyOptions({ width: chartContainerRef.current.clientWidth, height: chartContainerRef.current.clientHeight });
        };
        window.addEventListener('resize', handleResize);
        
        const series = chart.addCandlestickSeries({
            upColor: '#34d399', downColor: '#f87171', borderVisible: false, wickUpColor: '#34d399', wickDownColor: '#f87171',
        });
        series.setData(chartData);

        chartRef.current = chart;
        seriesRef.current = series;

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

        if (config.ema9.show) {
            const ema9 = chart.addLineSeries({ color: config.ema9.color, lineWidth: config.ema9.lineWidth as any, title: 'EMA 9' });
            ema9.setData(calculateEMA(chartData, 9));
        }
        if (config.ema21.show) {
            const ema21 = chart.addLineSeries({ color: config.ema21.color, lineWidth: config.ema21.lineWidth as any, title: 'EMA 21' });
            ema21.setData(calculateEMA(chartData, 21));
        }

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
    }, [chartData, config.ema9.show, config.ema9.color, config.ema9.lineWidth, config.ema21.show, config.ema21.color, config.ema21.lineWidth]);

    const handleMapToChart = (clientX: number, clientY: number) => {
        if (!chartRef.current || !seriesRef.current || !chartContainerRef.current) return null;
        const rect = chartContainerRef.current.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        const logical = chartRef.current.timeScale().coordinateToLogical(x);
        const price = seriesRef.current.coordinateToPrice(y);
        return { logical, price, x, y };
    };

    const onMouseDown = (e: React.MouseEvent) => {
        if (activeTool === 'pointer') { setSelectedDrawing(null); return; }
        const mapped = handleMapToChart(e.clientX, e.clientY);
        if (!mapped || mapped.logical === null) return;

        if (activeTool === 'text') {
            const txt = prompt("Enter text overlay:");
            if (txt) setDrawings([...drawings, { id: Date.now(), type: 'text', l1: mapped.logical, p1: mapped.price, l2: mapped.logical, p2: mapped.price, txt, color: config.drawingColor }]);
            setActiveTool('pointer');
            return;
        }

        setIsDrawing(true);
        setCurrentDrawing({ 
            id: Date.now(), type: activeTool, 
            l1: mapped.logical, p1: mapped.price, l2: mapped.logical, p2: mapped.price,
            path: [{l: mapped.logical, p: mapped.price}], color: config.drawingColor 
        });
    };

    const onMouseMove = (e: React.MouseEvent) => {
        if (!isDrawing || !currentDrawing) return;
        const mapped = handleMapToChart(e.clientX, e.clientY);
        if (!mapped || mapped.logical === null) return;

        if (currentDrawing.type === 'pencil' || currentDrawing.type === 'patterns') {
            setCurrentDrawing({ ...currentDrawing, path: [...currentDrawing.path, {l: mapped.logical, p: mapped.price}] });
        } else {
            setCurrentDrawing({ ...currentDrawing, l2: mapped.logical, p2: mapped.price });
        }
    };

    const onMouseUp = () => {
        if (isDrawing && currentDrawing) {
            setDrawings([...drawings, currentDrawing]);
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
        { id: 'line', icon: <Slash size={16} /> },
        { id: 'pencil', icon: <Pencil size={16} /> },
        { id: 'text', icon: <Type size={16} /> },
        { id: 'fib', icon: <ListMinus size={16} /> },
    ];

    return (
        <div className="w-full relative border border-[#171717] rounded-xl overflow-visible bg-[#000000] flex flex-col" style={{ minHeight: '600px' }}>
            <div className="flex items-center justify-between px-3 py-2 border-b border-[#171717] bg-[#0A0A0A] relative z-50">
                <div className="flex items-center space-x-4">
                    <div className="text-[13px] font-semibold text-zinc-200">{symbol}</div>
                    <div className="flex items-center space-x-1 border-l border-[#171717] pl-3 overflow-x-auto no-scrollbar">
                        {TIMEFRAMES.map((tf) => (
                            <button key={tf.value} onClick={() => { setTimeframe(tf.value); setCursorDate(null); }} className={`px-2 py-1 rounded text-[11px] font-medium transition-colors ${timeframe === tf.value ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}`}>
                                {tf.label}
                            </button>
                        ))}
                        <div className="relative ml-2">
                            <button onClick={() => setShowDatePanel(!showDatePanel)} className="p-1.5 rounded transition-colors text-zinc-500 hover:text-zinc-300 ml-2">
                                <Search size={14} />
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
                    <div className="relative">
                        <button onClick={() => { setShowIndicators(!showIndicators); setShowSettings(false); }} className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-[#171717] text-zinc-300 text-xs hover:bg-[#27272a] transition-all">
                            <Activity size={14} /> <span>Indicators</span>
                        </button>
                        {showIndicators && (
                            <div className="absolute top-10 right-0 w-56 bg-[#0A0A0A] border border-[#171717] p-2 rounded-lg shadow-2xl z-50">
                                <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2 px-2">Moving Averages</div>
                                <div className="space-y-1">
                                    <label className="flex items-center px-2 py-1.5 hover:bg-[#171717] rounded cursor-pointer text-xs text-zinc-300">
                                        <input type="checkbox" checked={config.ema9.show} onChange={(e) => setConfig({...config, ema9: {...config.ema9, show: e.target.checked}})} className="mr-2 accent-[#38BDF8]" /> EMA 9
                                    </label>
                                    <label className="flex items-center px-2 py-1.5 hover:bg-[#171717] rounded cursor-pointer text-xs text-zinc-300">
                                        <input type="checkbox" checked={config.ema21.show} onChange={(e) => setConfig({...config, ema21: {...config.ema21, show: e.target.checked}})} className="mr-2 accent-[#FCD34D]" /> EMA 21
                                    </label>
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
                    {tools.map(tool => (
                        <button key={tool.id} title={tool.id.toUpperCase()} onClick={() => setActiveTool(tool.id)} className={`p-2 rounded-md transition-colors ${activeTool === tool.id ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}`}>
                            {tool.icon}
                        </button>
                    ))}
                    {drawings.length > 0 && (
                        <button onClick={() => setDrawings([])} className="p-2 mt-4 rounded-md text-xs font-bold text-rose-500 hover:bg-[#171717] transition-all" title="Clear All">
                            ?
                        </button>
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
                                    <g key={d.id} style={{ pointerEvents: activeTool === 'pointer' ? 'auto' : 'none' }}>
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={d.color || config.drawingColor} strokeWidth="3" cursor="pointer" onClick={(e)=>{e.stopPropagation(); setSelectedDrawing(d.id);}} />
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" />}
                                    </g>
                                );
                            }
                            if (d.type === 'fib') {
                                const p1 = getCoordinate(d.l1, d.p1); const p2 = getCoordinate(d.l2, d.p2);
                                const diff = d.p2 - d.p1;
                                const fibLevels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
                                return (
                                    <g key={d.id} style={{ pointerEvents: activeTool === 'pointer' ? 'auto' : 'none' }} cursor="pointer" onClick={(e)=>{e.stopPropagation(); setSelectedDrawing(d.id);}}>
                                        <line x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={config.fibColor} strokeDasharray="4" opacity={config.fibOpacity} strokeWidth="2" />
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
                                        {selectedDrawing === d.id && <circle cx={p1.x} cy={p1.y} r="6" fill="#ffffff" />}
                                        {selectedDrawing === d.id && <circle cx={p2.x} cy={p2.y} r="6" fill="#ffffff" />}
                                    </g>
                                )
                            }
                            if (d.type === 'pencil' || d.type === 'patterns') {
                                const pts = d.path.map((p:any) => getCoordinate(p.l, p.p));
                                const dPath = pts.map((p:any, i:number) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');        
                                return <path key={d.id} d={dPath} fill="transparent" stroke={d.color || config.drawingColor} strokeWidth="3" style={{ pointerEvents: activeTool === 'pointer' ? 'auto' : 'none' }} cursor="pointer" onClick={(e)=>{e.stopPropagation(); setSelectedDrawing(d.id);}} />;
                            }
                            if (d.type === 'text') {
                                const p = getCoordinate(d.l1, d.p1);
                                return <text key={d.id} x={p.x} y={p.y} fill={d.color || config.drawingColor} style={{ pointerEvents: activeTool === 'pointer' ? 'auto' : 'none' }} fontSize="16" cursor="pointer" onClick={(e)=>{e.stopPropagation(); setSelectedDrawing(d.id);}} fontFamily="monospace">{d.txt}</text>;
                            }
                            return null;
                        })}

                        {isDrawing && currentDrawing && currentDrawing.type === 'line' && (
                            <line x1={getCoordinate(currentDrawing.l1, currentDrawing.p1).x} y1={getCoordinate(currentDrawing.l1, currentDrawing.p1).y} x2={getCoordinate(currentDrawing.l2, currentDrawing.p2).x} y2={getCoordinate(currentDrawing.l2, currentDrawing.p2).y} stroke={config.drawingColor} strokeWidth="2" />
                        )}
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
                            <input type="color" title="Change Color" value={drawings.find(d=>d.id===selectedDrawing)?.color || config.drawingColor} onChange={(e) => setDrawings(drawings.map(d => d.id === selectedDrawing ? {...d, color: e.target.value} : d))} className="w-5 h-5 rounded cursor-pointer" style={{background: 'transparent', border: 0, padding: 0}} />
                            <div className="h-3 w-px bg-[#171717] mx-1"></div>
                            <button onClick={() => { setDrawings(drawings.filter(d => d.id !== selectedDrawing)); setSelectedDrawing(null); }} className="p-1 hover:bg-[#171717] rounded text-rose-500" title="Delete"><Trash2 size={16}/></button>
                            <button onClick={() => setSelectedDrawing(null)} className="p-1 hover:bg-[#171717] rounded text-zinc-400"><X size={16}/></button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
"""

with open("components/TradingChart.tsx", "w", encoding="utf-8") as f:
    f.write(code)

print("SUCCESSFULLY WROTE PYTHON")
