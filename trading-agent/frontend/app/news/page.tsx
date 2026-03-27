"use client";

import { useState, useEffect } from "react";
import { fetchFromAPI } from "@/lib/api";
import { NewsImpact } from "@/lib/types";
import { Button } from "@/components/ui/button";

export default function NewsPage() {
    const [news, setNews] = useState<NewsImpact[]>([]);
    const [filter, setFilter] = useState<"ALL" | "SIGNIFICANT" | "SEVERE">("ALL");

    useEffect(() => {
        fetchFromAPI("/news").then(data => setNews(data)).catch(console.error);
    }, []);

    const filtered = news.filter(n => filter === "ALL" || n.severity === filter);

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-white">News Analysis</h1>
                <div className="flex space-x-2">
                    <Button variant={filter === "ALL" ? "default" : "outline"} size="sm" onClick={() => setFilter("ALL")}>All</Button>
                    <Button variant={filter === "SIGNIFICANT" ? "default" : "outline"} size="sm" onClick={() => setFilter("SIGNIFICANT")}>Significant</Button>
                    <Button variant={filter === "SEVERE" ? "destructive" : "outline"} size="sm" onClick={() => setFilter("SEVERE")}>Severe</Button>
                </div>
            </div>
            
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead className="uppercase tracking-wider border-b border-zinc-800 bg-zinc-950/50 text-zinc-400">
                            <tr>
                                <th className="px-6 py-4">Severity</th>
                                <th className="px-6 py-4">Asset / Direction</th>
                                <th className="px-6 py-4">Trust Score</th>
                                <th className="px-6 py-4 w-full">Rationale</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/50">
                            {filtered.length > 0 ? (
                                filtered.map((n, i) => (
                                    <tr key={i} className="hover:bg-zinc-800/20 transition-colors">
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 text-xs font-bold rounded-md ${
                                                n.severity === 'SEVERE' ? 'bg-red-500/20 text-red-500' : 
                                                n.severity === 'SIGNIFICANT' ? 'bg-orange-500/20 text-orange-500' : 
                                                'bg-zinc-500/20 text-zinc-400'
                                            }`}>
                                                {n.severity}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="font-medium text-white">{n.asset}</div>
                                            <div className={`text-xs ${n.direction === 'up' ? 'text-green-500' : n.direction === 'down' ? 'text-red-500' : 'text-zinc-500'}`}>
                                                {n.direction.toUpperCase()} ({n.magnitude_pct_high}%)
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-zinc-300">{(n.trust_score * 100).toFixed(0)}%</td>
                                        <td className="px-6 py-4 text-zinc-400 whitespace-normal max-w-[400px]">
                                            {n.rationale}
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colSpan={4} className="px-6 py-12 text-center text-zinc-500">
                                        No recent news analysis.
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