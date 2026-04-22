
"use client";

import React, { useEffect, useRef, useState } from "react";
import { createChart, IChartApi, ISeriesApi, ColorType, IPriceLine, LineStyle } from "lightweight-charts";
import { Search, Pencil, Type, Activity, MousePointer2, Slash, Settings, Trash2, ListMinus, X, Ruler, ArrowRightToLine, Palette, Undo, Redo, Eraser, Plus, Minus, Brain } from "lucide-react";
import { subscribeToLiveWs } from "@/lib/api";

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

export default function TradingChart({ symbol = "BTCUSDT", currencyRate = 1, currencyPrefix = "$" }: { symbol?: string, currencyRate?: number, currencyPrefix?: string }) {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const predictionSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
    const buyPriceLineRef = useRef<IPriceLine | null>(null);
    const sellPriceLineRef = useRef<IPriceLine | null>(null);
    const ema9Ref = useRef<ISeriesApi<"Line"> | null>(null);
    const ema21Ref = useRef<ISeriesApi<"Line"> | null>(null);

    const [chartData, setChartData] = useState<any[]>([]);
        const [timeframe, setTimeframe] = useState("15m");
    const [cursorDate, setCursorDate] = useState<number | null>(null);

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

        // WebSocket Live Updates
    useEffect(() => {
        if (cursorDate || !symbol) return; // Do not live update historical views
        let fetchTimeframe = timeframe;
        if (timeframe === '3M' || timeframe === '1Y' || timeframe === 'ALL') fetchTimeframe = '1M';
        
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

                    // Dynamically recalculate EMA tails
                    if (config.ema9.show && ema9Ref.current) {
                         const emaTails = calculateEMA(chartDataRef.current, 9);
                         if (emaTails.length > 0) ema9Ref.current.update(emaTails[emaTails.length - 1] as any);
                    }
                    if (config.ema21.show && ema21Ref.current) {
                         const emaTails = calculateEMA(chartDataRef.current, 21);
                         if (emaTails.length > 0) ema21Ref.current.update(emaTails[emaTails.length - 1] as any);
                    }
                }
            }
        };

        return () => ws.close();
    }, [symbol, timeframe, cursorDate, config.ema9.show, config.ema21.show, currencyRate]);

    // Backend Live Updates for Predictions
    useEffect(() => {
        const liveWs = subscribeToLiveWs((topic, data) => {
            if (topic === "prediction_update") {
                 setLastPredictionData(data);
            }
        });
        
        return () => liveWs.close();
    }, []);

    useEffect(() => {
        if (!lastPredictionData || !predictionSeriesRef.current) return;
        const data = lastPredictionData;

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
             if (seriesRef.current) {
                if (buyPriceLineRef.current) seriesRef.current.removePriceLine(buyPriceLineRef.current);
                if (sellPriceLineRef.current) seriesRef.current.removePriceLine(sellPriceLineRef.current);
                buyPriceLineRef.current = null;
                sellPriceLineRef.current = null;
             }
        }
    }, [lastPredictionData, showPredictions, timeframe, currencyRate]);

    // Fetch Data
    useEffect(() => {
        hasMoreHistory.current = true;
        isFetchingHistory.current = false;
        let fetchTimeframe = timeframe;
        if (timeframe === "3M" || timeframe === "1Y" || timeframe === "ALL") fetchTimeframe = "1M";
        
        let url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${fetchTimeframe}&limit=1000`;
        if (cursorDate) url += `&startTime=${cursorDate}`;

        fetch(url)
            .then(res => res.json())
            .then((data: any[]) => {
                const formatted = data.map((d: any) => ({
                    time: d[0] / 1000, 
                    open: parseFloat(d[1]) * currencyRate, 
                    high: parseFloat(d[2]) * currencyRate, 
                    low: parseFloat(d[3]) * currencyRate, 
                    close: parseFloat(d[4]) * currencyRate,
                }));
                const uniqueData = formatted.filter((v: any, i: number, a: any[]) => a.findIndex((t: any) => (t.time === v.time)) === i);
                chartDataRef.current = uniqueData;
                setChartData(uniqueData);
            }).catch(console.error);
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
            const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${fetchTimeframe}&limit=1000&endTime=${earliestTime * 1000 - 1}`;
            
            fetch(url)
                .then(res => res.json())
                .then(data => {
                    if (data.length === 0) {
                        hasMoreHistory.current = false;
                        isFetchingHistory.current = false;
                        return;
                    }
                    const formatted = data.map((d: any) => ({
                        time: d[0] / 1000, 
                        open: parseFloat(d[1]) * currencyRate, 
                        high: parseFloat(d[2]) * currencyRate, 
                        low: parseFloat(d[3]) * currencyRate, 
                        close: parseFloat(d[4]) * currencyRate,
                    }));
                    
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
                if (logicalRange.from < -50) {
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

        series.setData(chartData);
        predictionSeriesRef.current = predictionSeries;

        chartRef.current = chart;
        seriesRef.current = series;

        // Force a resize slightly after mount to ensure layout calculation
        setTimeout(() => {
            if (chartContainerRef.current) {
                chart.applyOptions({ width: chartContainerRef.current.clientWidth });
                chart.timeScale().fitContent();
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

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
        };
        }, [symbol, config.ema9.show, config.ema9.color, config.ema9.lineWidth, config.ema21.show, config.ema21.color, config.ema21.lineWidth, isVibrantColors, currencyPrefix]);

    useEffect(() => {
        if (chartData.length > 0 && seriesRef.current) {
            seriesRef.current.setData(chartData);
            
// Update EMAs
            if (ema9Ref.current) ema9Ref.current.setData(calculateEMA(chartData, 9));
            if (ema21Ref.current) ema21Ref.current.setData(calculateEMA(chartData, 21));
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
            <div className="flex items-center justify-between px-3 py-2 border-b border-[#171717] bg-[#0A0A0A] relative z-50">
                <div className="flex items-center space-x-4">
                    <div className="text-[13px] font-semibold text-zinc-200">{symbol}</div>
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
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">SMA 50</span><span className="text-xs text-zinc-500 mt-0.5">Simple Moving Average 50</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">SMA 200</span><span className="text-xs text-zinc-500 mt-0.5">Simple Moving Average 200</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">VWMA</span><span className="text-xs text-zinc-500 mt-0.5">Volume Weighted Moving Average</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Ichimoku Cloud</span><span className="text-xs text-zinc-500 mt-0.5">Ichimoku Kinko Hyo</span></div>
                                                </label>
                                            </div>
                                        </div>
                                        <div className="px-4 pb-4">
                                            <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-4 px-2">Oscillators</div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">RSI</span><span className="text-xs text-zinc-500 mt-0.5">Relative Strength Index</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">MACD</span><span className="text-xs text-zinc-500 mt-0.5">Moving Average Convergence Divergence</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Stochastic</span><span className="text-xs text-zinc-500 mt-0.5">Stochastic Oscillator</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">CCI</span><span className="text-xs text-zinc-500 mt-0.5">Commodity Channel Index</span></div>
                                                </label>
                                                 <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Awesome Oscillator</span><span className="text-xs text-zinc-500 mt-0.5">AO</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Momentum</span><span className="text-xs text-zinc-500 mt-0.5">Momentum Indicator</span></div>
                                                </label>
                                            </div>
                                        </div>
                                        <div className="px-4 pb-4">
                                            <div className="text-xs font-semibold text-neutral-500 uppercase tracking-wider mb-4 px-2">Volatility & Volume</div>
                                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Bollinger Bands</span><span className="text-xs text-zinc-500 mt-0.5">Bollinger Bands</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">ATR</span><span className="text-xs text-zinc-500 mt-0.5">Average True Range</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">Volume</span><span className="text-xs text-zinc-500 mt-0.5">Trading Volume</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">VWAP</span><span className="text-xs text-zinc-500 mt-0.5">Volume Weighted Average Price</span></div>
                                                </label>
                                                <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
                                                    <div className="flex flex-col"><span className="text-neutral-200 font-medium group-hover:text-white transition-colors">OBV</span><span className="text-xs text-zinc-500 mt-0.5">On Balance Volume</span></div>
                                                </label>
                                                 <label className="flex items-start px-3 py-3 hover:bg-[#17171A] rounded-xl cursor-pointer text-sm text-zinc-300 transition-colors group">
                                                    <input type="checkbox" className="mt-1 mr-4 w-4 h-4 accent-zinc-500 rounded bg-[#1A1A1D] border-neutral-700" />
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
                                <button key={tool.id} title={tool.id.toUpperCase()} onClick={() => setActiveTool(tool.id)} className={`p-2 rounded-md transition-colors ${activeTool === tool.id ? 'bg-[#171717] text-zinc-200' : 'text-zinc-500 hover:text-zinc-300'}`}>
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
