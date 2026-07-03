"use client";

import { useAppState } from "@/lib/context";

export default function PositionsPage() {
    const { positions } = useAppState();

    const longs = positions.filter(p => p.direction === "long").length;
    const shorts = positions.length - longs;
    const netPnl = positions.reduce((s, p) => s + (p.unrealised_pnl || 0), 0);

    return (
        <div className="mx-auto max-w-[1400px] space-y-6">
            <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight text-white">Live Positions</h1>
                    <p className="mt-1 text-sm text-zinc-500">Open exposure across every connected venue, marked in real time.</p>
                </div>
                <div className="flex items-center gap-6 text-sm">
                    <div className="text-right">
                        <div className="text-[10px] uppercase tracking-widest text-zinc-600">Open</div>
                        <div className="font-mono text-zinc-200">{positions.length} <span className="text-zinc-600">·</span> {longs}L / {shorts}S</div>
                    </div>
                    <div className="text-right">
                        <div className="text-[10px] uppercase tracking-widest text-zinc-600">Unrealised PnL</div>
                        <div className={`font-mono font-medium ${netPnl >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                            {netPnl >= 0 ? "+" : ""}${netPnl.toFixed(2)}
                        </div>
                    </div>
                </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-[#171717] bg-[#0A0A0A]">
                <div className="overflow-x-auto">
                    <table className="w-full whitespace-nowrap text-left text-sm">
                        <thead className="border-b border-[#171717] bg-black/40 text-[10px] uppercase tracking-widest text-zinc-500">
                            <tr>
                                <th className="px-6 py-4 font-medium">Asset</th>
                                <th className="px-6 py-4 font-medium">Side</th>
                                <th className="px-6 py-4 font-medium">Size (USD)</th>
                                <th className="px-6 py-4 font-medium">Entry</th>
                                <th className="px-6 py-4 font-medium">Current</th>
                                <th className="px-6 py-4 font-medium">PnL</th>
                                <th className="px-6 py-4 text-right font-medium">SL / TP</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[#141416]">
                            {positions.length > 0 ? (
                                positions.map((pos, i) => {
                                    const isProfit = (pos.unrealised_pnl || 0) >= 0;
                                    const isLong = pos.direction === "long";
                                    return (
                                        <tr key={pos.id || i} className="transition-colors hover:bg-white/[0.015]">
                                            <td className="px-6 py-4 font-medium text-white">{pos.asset}</td>
                                            <td className="px-6 py-4">
                                                <span className={`rounded-md px-2 py-1 text-[11px] font-semibold ${isLong ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                                                    {pos.direction.toUpperCase()}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 font-mono text-zinc-300">${pos.size_usd.toFixed(2)}</td>
                                            <td className="px-6 py-4 font-mono text-zinc-300">${pos.entry_price.toFixed(4)}</td>
                                            <td className="px-6 py-4 font-mono text-zinc-300">${pos.current_price?.toFixed(4) || "…"}</td>
                                            <td className={`px-6 py-4 font-mono font-medium ${isProfit ? 'text-emerald-400' : 'text-rose-400'}`}>
                                                {isProfit ? "+" : ""}${(pos.unrealised_pnl || 0).toFixed(2)}
                                            </td>
                                            <td className="px-6 py-4 text-right font-mono text-zinc-400">
                                                <span className="text-rose-400/80">${pos.stop_loss.toFixed(4)}</span>
                                                <span className="text-zinc-600"> / </span>
                                                <span className="text-emerald-400/80">${pos.take_profit.toFixed(4)}</span>
                                            </td>
                                        </tr>
                                    );
                                })
                            ) : (
                                <tr>
                                    <td colSpan={7} className="px-6 py-16 text-center text-zinc-600">
                                        No open positions.
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
