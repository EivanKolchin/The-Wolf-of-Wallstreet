"use client";

import { useAppState } from "@/lib/context";

export default function PositionsPage() {
    const { positions } = useAppState();

    return (
        <div className="space-y-6">
            <h1 className="text-2xl font-bold text-white">Live Positions</h1>
            
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead className="uppercase tracking-wider border-b border-zinc-800 bg-zinc-950/50 text-zinc-400">
                            <tr>
                                <th className="px-6 py-4">Asset</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4">Size (USD)</th>
                                <th className="px-6 py-4">Entry</th>
                                <th className="px-6 py-4">Current</th>
                                <th className="px-6 py-4">PnL</th>
                                <th className="px-6 py-4 text-right">SL / TP</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/50">
                            {positions.length > 0 ? (
                                positions.map((pos, i) => {
                                    const isProfit = (pos.unrealised_pnl || 0) >= 0;
                                    return (
                                        <tr key={pos.id || i} className="hover:bg-zinc-800/20 transition-colors">
                                            <td className="px-6 py-4 font-medium text-white">{pos.asset}</td>
                                            <td className="px-6 py-4">
                                                <span className={`px-2 py-1 text-xs font-bold rounded-md ${pos.direction === 'long' ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'}`}>
                                                    {pos.direction.toUpperCase()}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 text-zinc-300">${pos.size_usd.toFixed(2)}</td>
                                            <td className="px-6 py-4 text-zinc-300">${pos.entry_price.toFixed(4)}</td>
                                            <td className="px-6 py-4 text-zinc-300">${pos.current_price?.toFixed(4) || "..."}</td>
                                            <td className={`px-6 py-4 font-medium ${isProfit ? 'text-green-500' : 'text-red-500'}`}>
                                                {isProfit ? "+" : ""}${(pos.unrealised_pnl || 0).toFixed(2)}
                                            </td>
                                            <td className="px-6 py-4 text-right text-zinc-400">
                                                <span className="text-red-400">${pos.stop_loss.toFixed(4)}</span> / <span className="text-green-400">${pos.take_profit.toFixed(4)}</span>
                                            </td>
                                        </tr>
                                    );
                                })
                            ) : (
                                <tr>
                                    <td colSpan={7} className="px-6 py-12 text-center text-zinc-500">
                                        No open positions
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
