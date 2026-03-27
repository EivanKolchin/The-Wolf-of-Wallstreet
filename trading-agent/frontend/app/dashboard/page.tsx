"use client";

import { useAppState } from "@/lib/context";
import TradingChart from "@/components/TradingChart";
import { ArrowDownIcon, ArrowUpIcon, Activity } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function DashboardPage() {
    const { portfolio, signals, news, status } = useAppState();

    const pnl = portfolio?.daily_pnl || 0;
    const isPnlPositive = pnl >= 0;

    const latestSignal = signals[0];
    const confidence = latestSignal ? latestSignal.confidence : 0;

    return (
        <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* PnL Card */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex flex-col justify-between">
                    <span className="text-zinc-400 text-sm font-medium">Total Portfolio Value</span>
                    <div className="mt-2 flex items-baseline space-x-2">
                        <span className="text-3xl font-bold text-white">${portfolio?.total_value_usd?.toFixed(2) || "0.00"}</span>
                    </div>
                    <div className="mt-4 flex items-center justify-between text-sm">
                        <span className="text-zinc-500">Daily PnL</span>
                        <div className={`flex items-center font-medium ${isPnlPositive ? 'text-green-500' : 'text-red-500'}`}>
                            {isPnlPositive ? <ArrowUpIcon className="w-4 h-4 mr-1" /> : <ArrowDownIcon className="w-4 h-4 mr-1" />}
                            ${Math.abs(pnl).toFixed(2)}
                        </div>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-sm">
                        <span className="text-zinc-500">Drawdown</span>
                        <span className="text-zinc-300">{(portfolio?.drawdown_pct || 0).toFixed(2)}%</span>
                    </div>
                </div>

                {/* Confidence Gauge */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex flex-col items-center justify-center relative overflow-hidden">
                    <span className="text-zinc-400 text-sm font-medium absolute top-4 left-4">NN Confidence</span>
                    <div className="relative w-32 h-32 mt-6 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90">
                            <circle cx="64" cy="64" r="56" className="stroke-zinc-800" strokeWidth="12" fill="none" />
                            <motion.circle
                                cx="64" cy="64" r="56"
                                className="stroke-blue-500"
                                strokeWidth="12" fill="none"
                                strokeDasharray="351.858"
                                strokeDashoffset={351.858 - (351.858 * (confidence * 100)) / 100}
                                strokeLinecap="round"
                                initial={{ strokeDashoffset: 351.858 }}
                                animate={{ strokeDashoffset: 351.858 - (351.858 * (confidence * 100)) / 100 }}
                                transition={{ duration: 1, ease: "easeOut" }}
                            />
                        </svg>
                        <div className="absolute flex flex-col items-center">
                            <span className="text-2xl font-bold text-white">{(confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                </div>

                {/* Regime Widget */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex flex-col justify-between">
                    <span className="text-zinc-400 text-sm font-medium">Market Regime</span>
                    <div className="flex flex-col items-center mt-4">
                        <Activity className="w-10 h-10 text-zinc-600 mb-2" />
                        <span className="text-xl font-bold text-white capitalize">{latestSignal?.direction || "NEUTRAL"}</span>
                    </div>
                    <div className="mt-6">
                        <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
                            <motion.div 
                                className={`h-full ${latestSignal?.direction === 'long' ? 'bg-green-500' : latestSignal?.direction === 'short' ? 'bg-red-500' : 'bg-zinc-500'}`} 
                                initial={{ width: 0 }}
                                animate={{ width: `${confidence * 100}%` }}
                            />
                        </div>
                    </div>
                </div>

                {/* Active News Widget */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex flex-col justify-between relative overflow-hidden group">
                    <span className="text-zinc-400 text-sm font-medium">Active News Impact</span>
                    {news ? (
                        <div className="mt-4">
                            <div className="flex items-center space-x-2">
                                <span className={`px-2 py-1 text-xs font-bold rounded-md ${
                                    news.severity === 'SEVERE' ? 'bg-red-500/20 text-red-500' : 
                                    news.severity === 'SIGNIFICANT' ? 'bg-orange-500/20 text-orange-500' : 
                                    'bg-zinc-500/20 text-zinc-400'
                                }`}>
                                    {news.severity}
                                </span>
                                <span className="text-sm font-semibold text-white">{news.asset}</span>
                            </div>
                            <p className="mt-3 text-sm text-zinc-300 line-clamp-3">{news.rationale}</p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-zinc-500 mt-4">
                            <span className="text-sm">No significant active news.</span>
                        </div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                    <TradingChart symbol="BTCUSDT" />
                </div>
                
                {/* Signal Feed */}
                <div className="bg-zinc-900 border border-zinc-800 rounded-xl flex flex-col h-[400px]">
                    <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                        <span className="font-semibold text-white">Live Signal Feed</span>
                        <span className="text-xs text-zinc-500">Auto-updating</span>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-2">
                        <AnimatePresence>
                            {signals.map((sig, idx) => (
                                <motion.div
                                    key={idx}
                                    initial={{ opacity: 0, y: -20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0 }}
                                    className="bg-zinc-950/50 border border-zinc-800/50 p-3 rounded-lg flex items-center justify-between"
                                >
                                    <div className="flex items-center space-x-3">
                                        <div className={`w-2 h-2 rounded-full ${sig.direction === 'long' ? 'bg-green-500' : sig.direction === 'short' ? 'bg-red-500' : 'bg-zinc-500'}`} />
                                        <div>
                                            <div className="text-sm font-semibold text-white uppercase">{sig.asset} {sig.direction}</div>
                                            <div className="text-xs text-zinc-500">{new Date(sig.timestamp || Date.now()).toLocaleTimeString()}</div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-sm font-medium text-white">{(sig.confidence * 100).toFixed(1)}%</div>
                                        <div className="text-xs text-zinc-500">Confidence</div>
                                    </div>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        {signals.length === 0 && (
                            <div className="text-zinc-500 text-sm text-center mt-10">Waiting for signals...</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
