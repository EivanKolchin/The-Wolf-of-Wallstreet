"use client";

import { useState, useEffect } from "react";
import { fetchFromAPI } from "@/lib/api";
import { Trade } from "@/lib/types";

export default function AuditPage() {
    const [trades, setTrades] = useState<Trade[]>([]);

    useEffect(() => {
        fetchFromAPI("/trades").then(data => setTrades(data)).catch(console.error);
    }, []);

    return (
        <div className="space-y-6">
            <div className="flex flex-col space-y-1">
                <h1 className="text-2xl font-bold text-white">On-Chain Audit Log</h1>
                <p className="text-zinc-500 text-sm">Every resolved epoch is committed permanently to Kite AI Chain.</p>
            </div>
            
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead className="uppercase tracking-wider border-b border-zinc-800 bg-zinc-950/50 text-zinc-400">
                            <tr>
                                <th className="px-6 py-4">Trade ID</th>
                                <th className="px-6 py-4">Asset</th>
                                <th className="px-6 py-4">Action</th>
                                <th className="px-6 py-4">Pred. Score</th>
                                <th className="px-6 py-4">Kite Chain Tx</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/50">
                            {trades.length > 0 ? (
                                trades.map((t, i) => (
                                    <tr key={i} className="hover:bg-zinc-800/20 transition-colors">
                                        <td className="px-6 py-4 font-mono text-zinc-400">{t.id.slice(0, 8)}</td>
                                        <td className="px-6 py-4 font-medium text-white">{t.asset}</td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 text-xs font-bold rounded-md ${t.direction === 'long' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}`}>
                                                {t.direction.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-zinc-300">
                                            {t.prediction_score !== undefined ? `${(t.prediction_score * 100).toFixed(1)}%` : "N/A"}
                                        </td>
                                        <td className="px-6 py-4 text-blue-500 hover:text-blue-400 underline decoration-blue-500/30 underline-offset-4">
                                            {t.kite_tx_hash ? (
                                                <a href={`https://testnet.kitescan.ai/tx/${t.kite_tx_hash}`} target="_blank" rel="noreferrer">
                                                    {t.kite_tx_hash.slice(0, 6)}...{t.kite_tx_hash.slice(-4)}
                                                </a>
                                            ) : (
                                                <span className="text-zinc-600 no-underline">Pending</span>
                                            )}
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={5} className="px-6 py-12 text-center text-zinc-500">
                                        No trades audited yet.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
