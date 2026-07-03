"use client";

import { useState, useEffect } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { fetchFromAPI } from "@/lib/api";
import { NewsImpact } from "@/lib/types";
import { NewsScannerWidget } from "@/components/NewsScannerWidget";
import { LLM_FEEDBACK_LEVELS, newsFeedbackKey, severityClass, useNewsFeedback } from "@/lib/hooks/useNewsFeedback";

const FILTERS = ["ALL", "MILD", "SIGNIFICANT", "SEVERE"] as const;
type Filter = typeof FILTERS[number];

type AgentHealth = { news_alive?: boolean; nn_alive?: boolean };

export default function NewsPage() {
    const [news, setNews] = useState<NewsImpact[]>([]);
    const [rawNews, setRawNews] = useState<any[]>([]);
    const [filter, setFilter] = useState<Filter>("ALL");
    const [health, setHealth] = useState<AgentHealth | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const { getRating, submitFeedback } = useNewsFeedback();

    useEffect(() => {
        let mounted = true;
        const load = async () => {
            try {
                const [recent, raw, h] = await Promise.all([
                    fetchFromAPI("/news/recent"),
                    fetchFromAPI("/news/raw"),
                    fetchFromAPI("/health").catch(() => null),
                ]);
                if (!mounted) return;
                setNews(Array.isArray(recent) ? recent : []);
                setRawNews(Array.isArray(raw) ? raw : []);
                setHealth(h);
                setLoadError(null);
            } catch (e: any) {
                if (mounted) setLoadError(e?.message || "Failed to load intelligence feed");
            }
        };
        load();
        const timer = setInterval(load, 5000);
        return () => {
            mounted = false;
            clearInterval(timer);
        };
    }, []);

    const filtered = news.filter(n => filter === "ALL" || n.severity === filter);

    const sendSeverityFeedback = async (pred: any, selectedValue: string, index: number) => {
        const key = newsFeedbackKey(pred, index);
        await submitFeedback(key, selectedValue as any, {
            feedback_type: "severity",
            prediction_id: pred.id,
            article_hash: pred.article_hash,
            headline: pred.headline,
            source_domain: pred.source_domain,
            asset: pred.asset,
            severity: pred.severity,
            selected_value: selectedValue,
        });
    };

    const trustScore = (n: any) => {
        const v = n.trust_score ?? n.trust_score_at_time;
        return typeof v === "number" && Number.isFinite(v) ? v : 0;
    };

    return (
        <div className="mx-auto max-w-[1400px] space-y-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-semibold tracking-tight text-white">News Intelligence</h1>
                    <p className="mt-1 text-sm text-zinc-500">
                        LLM-classified market events with directional impact and trust scoring.
                        {health && (
                            <span className="ml-2 font-mono text-[11px] text-zinc-600">
                                news agent {health.news_alive ? "live" : "offline"}
                            </span>
                        )}
                    </p>
                </div>
                <div className="inline-flex items-center gap-1 rounded-lg border border-[#1a1a1c] bg-[#0a0a0a] p-1">
                    {FILTERS.map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={`rounded-md px-3 py-1.5 text-[12px] font-medium capitalize tracking-wide transition-colors ${
                                filter === f ? 'bg-zinc-100 text-zinc-900' : 'text-zinc-500 hover:text-zinc-200'
                            }`}
                        >
                            {f.toLowerCase()}
                        </button>
                    ))}
                </div>
            </div>

            {loadError && (
                <div className="rounded-xl border border-rose-500/20 bg-rose-500/5 px-4 py-3 text-sm text-rose-300">
                    {loadError}
                </div>
            )}

            <div className="grid gap-6 lg:grid-cols-3">
                <div className="lg:col-span-1">
                    <NewsScannerWidget rawNews={rawNews} />
                </div>
                <div className="lg:col-span-2 rounded-2xl border border-[#171717] bg-[#0A0A0A] p-4 text-sm text-zinc-500">
                    {health && !health.news_alive ? (
                        <p>
                            The LLM news agent is not running. Start the full stack via <code className="text-zinc-400">python main.py</code> and ensure Ollama or your configured AI provider is set up in Settings.
                        </p>
                    ) : rawNews.length > 0 && filtered.length === 0 ? (
                        <p>
                            Scanner is receiving headlines but the LLM has not classified any as MILD, SIGNIFICANT, or SEVERE yet. Use thumbs on scanner items to steer what gets analysed.
                        </p>
                    ) : (
                        <p>
                            Rate severity with thumbs to calibrate future LLM classifications. Your ratings persist across page reloads. Scanner thumbs steer which headlines get sent to the LLM.
                        </p>
                    )}
                </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-[#171717] bg-[#0A0A0A]">
                <div className="overflow-x-auto">
                    <table className="w-full whitespace-nowrap text-left text-sm">
                        <thead className="border-b border-[#171717] bg-black/40 text-[10px] uppercase tracking-widest text-zinc-500">
                            <tr>
                                <th className="px-6 py-4 font-medium">Severity</th>
                                <th className="px-6 py-4 font-medium">Asset / Direction</th>
                                <th className="px-6 py-4 font-medium">Trust Score</th>
                                <th className="px-6 py-4 font-medium">Original Title</th>
                                <th className="w-full px-6 py-4 font-medium">LLM Summary</th>
                                <th className="px-6 py-4 font-medium">Classifier feedback</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-[#141416]">
                            {filtered.length > 0 ? (
                                filtered.map((n: any, i) => {
                                    const key = newsFeedbackKey(n, i);
                                    const selected = getRating(key);
                                    return (
                                    <tr key={key} className="transition-colors hover:bg-white/[0.015]">
                                        <td className="px-6 py-4">
                                            <span className={`rounded-md px-2 py-1 text-[11px] font-semibold uppercase tracking-wider ${severityClass(n.severity)}`}>
                                                {n.severity}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="font-medium text-white">{n.asset}</div>
                                            <div className={`font-mono text-xs ${n.direction === 'up' ? 'text-emerald-400' : n.direction === 'down' ? 'text-rose-400' : 'text-zinc-500'}`}>
                                                {String(n.direction).toUpperCase()} ({n.magnitude_pct_high}%)
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 font-mono text-zinc-300">{(trustScore(n) * 100).toFixed(0)}%</td>
                                        <td className="max-w-[260px] whitespace-normal px-6 py-4 text-[12px] text-zinc-300">
                                            <div className="line-clamp-2">{n.headline}</div>
                                        </td>
                                        <td className="max-w-[420px] whitespace-normal px-6 py-4 text-zinc-400">
                                            <div className="line-clamp-3 text-[11px]">{n.rationale}</div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex flex-wrap items-center gap-1.5">
                                                {LLM_FEEDBACK_LEVELS.map(level => (
                                                    <button
                                                        key={level}
                                                        type="button"
                                                        onClick={() => sendSeverityFeedback(n, level, i)}
                                                        className={`rounded-md border px-2 py-1 text-[10px] font-medium uppercase tracking-wide transition-colors ${
                                                            selected === level
                                                                ? 'border-zinc-300 bg-white/10 text-zinc-100'
                                                                : 'border-[#1f1f22] text-zinc-500 hover:border-zinc-500/40 hover:text-zinc-300'
                                                        }`}
                                                        title={`Classify similar future news as ${level.toLowerCase()}`}
                                                    >
                                                        {level}
                                                    </button>
                                                ))}
                                            </div>
                                        </td>
                                    </tr>
                                );
                                })
                            ) : (
                                <tr>
                                    <td colSpan={6} className="px-6 py-16 text-center text-zinc-600">
                                        {rawNews.length > 0
                                            ? "Scanner is active — waiting for LLM to flag market impacts."
                                            : "No recent news analysis. Ensure the news agent is running."}
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
