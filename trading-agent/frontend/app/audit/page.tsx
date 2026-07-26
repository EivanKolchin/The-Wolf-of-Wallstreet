"use client";

import { useState, useEffect } from "react";
import { API_BASE, fetchFromAPI } from "@/lib/api";
import { Trade } from "@/lib/types";

interface JournalEvent {
    ts: number;
    iso?: string;
    kind: "rebalance" | "order";
    symbol?: string;
    target_weight?: number;
    delta?: number;
    decision_price?: number;
    notional?: number;
    n_orders?: number;
    n_skipped?: number;
    gross?: number;
    net?: number;
    equity?: number;
    blocked?: string | null;
    paper?: boolean;
}

const cleanSym = (s: string) => s.replace("USDT", "").replace("-USD", "");
const fmtUsd = (n?: number) =>
    n == null ? "—" : `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export default function AuditPage() {
    const [events, setEvents] = useState<JournalEvent[]>([]);
    const [journalActive, setJournalActive] = useState<boolean | null>(null);
    const [trades, setTrades] = useState<Trade[]>([]);

    useEffect(() => {
        const load = () =>
            fetch(`${API_BASE}/strategy/journal?limit=200`)
                .then((r) => r.json())
                .then((d) => {
                    setJournalActive(!!d?.active);
                    setEvents(Array.isArray(d?.events) ? d.events : []);
                })
                .catch(() => setJournalActive(false));
        load();
        const id = setInterval(load, 8000);
        return () => clearInterval(id);
    }, []);

    // Trade-table fallback (live/broker mode), only fetched when the journal is inactive.
    useEffect(() => {
        if (journalActive === false) {
            fetchFromAPI("/trades").then((data) => setTrades(data)).catch(console.error);
        }
    }, [journalActive]);

    const showJournal = journalActive && events.length > 0;

    return (
        <div className="mx-auto max-w-[1400px] space-y-6">
            <div className="flex flex-col space-y-1">
                <h1 className="text-2xl font-semibold tracking-tight text-white">Trade Audit Log</h1>
                <p className="text-sm text-zinc-500">
                    {showJournal
                        ? "Every strategy-book rebalance and routed order, newest first (paper journal)."
                        : "Every resolved trade is recorded with a transaction statement (see the /statements folder)."}
                </p>
            </div>

            <div className="overflow-hidden rounded-2xl border border-[#171717] bg-[#0A0A0A]">
                <div className="overflow-x-auto">
                    {showJournal ? (
                        <table className="w-full whitespace-nowrap text-left text-sm">
                            <thead className="border-b border-[#171717] bg-black/40 text-[10px] uppercase tracking-widest text-zinc-500">
                                <tr>
                                    <th className="px-6 py-4 font-medium">Time</th>
                                    <th className="px-6 py-4 font-medium">Event</th>
                                    <th className="px-6 py-4 font-medium">Asset</th>
                                    <th className="px-6 py-4 font-medium">Detail</th>
                                    <th className="px-6 py-4 text-right font-medium">Notional / Equity</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-[#141416]">
                                {events.map((e, i) => {
                                    const when = e.iso
                                        ? new Date(e.iso).toLocaleString()
                                        : new Date(e.ts * 1000).toLocaleString();
                                    if (e.kind === "order") {
                                        const up = (e.delta ?? 0) >= 0;
                                        return (
                                            <tr key={i} className="transition-colors hover:bg-white/[0.015]">
                                                <td className="px-6 py-4 font-mono text-zinc-500">{when}</td>
                                                <td className="px-6 py-4">
                                                    <span className={`rounded-md px-2 py-1 text-[11px] font-semibold ${up ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                                        {up ? "BUY" : "SELL"}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 font-medium text-white">{e.symbol ? cleanSym(e.symbol) : "—"}</td>
                                                <td className="px-6 py-4 font-mono text-zinc-400 text-[11px]">
                                                    Δw {((e.delta ?? 0) * 100).toFixed(1)}% → target {((e.target_weight ?? 0) * 100).toFixed(1)}%
                                                    {e.decision_price != null && <span className="text-zinc-600"> @ {fmtUsd(e.decision_price)}</span>}
                                                </td>
                                                <td className="px-6 py-4 text-right font-mono text-zinc-300">{fmtUsd(e.notional)}</td>
                                            </tr>
                                        );
                                    }
                                    return (
                                        <tr key={i} className="transition-colors hover:bg-white/[0.015] bg-white/[0.01]">
                                            <td className="px-6 py-4 font-mono text-zinc-500">{when}</td>
                                            <td className="px-6 py-4">
                                                <span className="rounded-md bg-zinc-500/10 px-2 py-1 text-[11px] font-semibold text-zinc-300">REBALANCE</span>
                                            </td>
                                            <td className="px-6 py-4 text-zinc-600">—</td>
                                            <td className="px-6 py-4 font-mono text-zinc-400 text-[11px]">
                                                {e.n_orders ?? 0} orders · gross {((e.gross ?? 0) * 100).toFixed(0)}% · net {((e.net ?? 0) * 100).toFixed(0)}%
                                                {e.blocked && <span className="text-rose-400"> · blocked: {e.blocked}</span>}
                                            </td>
                                            <td className="px-6 py-4 text-right font-mono text-zinc-300">{fmtUsd(e.equity)}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    ) : (
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
                    )}
                </div>
            </div>
        </div>
    );
}
