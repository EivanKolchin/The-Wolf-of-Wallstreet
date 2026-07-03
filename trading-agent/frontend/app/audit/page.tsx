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
        <div className="mx-auto max-w-[1400px] space-y-6">
            <div className="flex flex-col space-y-1">
                <h1 className="text-2xl font-semibold tracking-tight text-white">Trade Audit Log</h1>
                <p className="text-sm text-zinc-500">Every resolved trade is recorded with a transaction statement (see the /statements folder).</p>
            </div>

            <div className="overflow-hidden rounded-2xl border border-[#171717] bg-[#0A0A0A]">
                <div className="overflow-x-auto">
                    <table className="w-full whitespace-nowrap text-left text-sm">
                        <thead className="border-b border-[#171717] bg-black/40 text-[10px] uppercase tracking-widest text-zinc-500">
                            <tr>
                                <th className="px-6 py-4 font-medium">Trade ID</th>
                                <th className="px-6 py-4 font-medium">Asset</th>
                                <th className="px-6 py-4 font-medium">Side</th>
                                <th className="px-6 py-4 font-medium">Pred. Score</th>
                                <th className="px-6 py-4 font-medium">P&amp;L (USD)</th>
                                <th className="px-6 py-4 font-medium">Rationale</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[#141416]">
                            {trades.length > 0 ? (
                                trades.map((t, i) => (
                                    <tr key={i} className="transition-colors hover:bg-white/[0.015]">
                                        <td className="px-6 py-4 font-mono text-zinc-500">{t.id.slice(0, 8)}</td>
                                        <td className="px-6 py-4 font-medium text-white">{t.asset}</td>
                                        <td className="px-6 py-4">
                                            <span className={`rounded-md px-2 py-1 text-[11px] font-semibold ${t.direction === 'long' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                                {t.direction.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 font-mono text-zinc-300">
                                            {t.prediction_score !== undefined ? `${(t.prediction_score * 100).toFixed(1)}%` : "N/A"}
                                        </td>
                                        <td className="px-6 py-4 font-mono">
                                            {t.pnl_usd !== undefined && t.pnl_usd !== null ? (
                                                <span className={t.pnl_usd >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                                                    {t.pnl_usd >= 0 ? '+' : ''}{t.pnl_usd.toFixed(2)}
                                                </span>
                                            ) : (
                                                <span className="text-zinc-600">Open</span>
                                            )}
                                        </td>
                                        <td className="max-w-md truncate px-6 py-4 text-[11px] text-zinc-400" title={t.rationale?.summary || ""}>
                                            {t.rationale?.summary || <span className="text-zinc-600">—</span>}
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={6} className="px-6 py-16 text-center text-zinc-600">
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
