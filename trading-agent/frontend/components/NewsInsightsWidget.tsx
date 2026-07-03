"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { LLM_FEEDBACK_LEVELS, newsFeedbackKey, severityClass, useNewsFeedback } from "@/lib/hooks/useNewsFeedback";

const FEEDBACK_TONE: Record<string, string> = {
  SEVERE: "border-rose-500/40 text-rose-300 hover:bg-rose-500/10",
  SIGNIFICANT: "border-amber-500/40 text-amber-300 hover:bg-amber-500/10",
  MILD: "border-sky-500/40 text-sky-300 hover:bg-sky-500/10",
  INSIGNIFICANT: "border-zinc-600 text-zinc-300 hover:bg-zinc-700/30",
};

export function NewsInsightsWidget({
  predictions,
  expanded = false,
}: { predictions: any[]; expanded?: boolean }) {
  const rows = predictions?.slice(0, 10) || [];
  const { getRating, submitFeedback } = useNewsFeedback();

  const sendSeverityFeedback = async (pred: any, targetSeverity: string, index: number) => {
    const key = newsFeedbackKey(pred, index);
    await submitFeedback(key, targetSeverity as any, {
      feedback_type: "severity",
      prediction_id: pred.id,
      article_hash: pred.article_hash,
      headline: pred.headline,
      source_domain: pred.source_domain,
      asset: pred.asset,
      severity: pred.severity,
      selected_value: targetSeverity,
    });
  };

  return (
    <Card className="flex h-full flex-col overflow-hidden">
      <CardHeader className="flex flex-row items-center justify-between border-b border-[#171717] pb-4">
        <CardTitle className="text-sm">LLM Intelligence Log</CardTitle>
        <span className="rounded-full border border-[#1f1f22] bg-[#0e0e10] px-2 py-0.5 font-mono text-[10px] text-zinc-500">
          {rows.length ? `${rows.length} calls` : "standby"}
        </span>
      </CardHeader>
      <CardContent className={`${expanded ? "max-h-[520px]" : "max-h-[320px]"} flex-1 overflow-y-auto p-0`}>
        <table className="w-full text-left">
          <thead className="sticky top-0 z-10 border-b border-[#171717] bg-[#0A0A0A] text-[10px] font-medium uppercase tracking-wider text-zinc-500">
            <tr>
              <th className="px-4 py-3 font-normal">Original Title</th>
              <th className="px-4 py-3 font-normal">LLM Summary</th>
              <th className="px-4 py-3 font-normal">Rating</th>
              <th className="px-4 py-3 font-normal">Prediction</th>
              <th className="px-4 py-3 font-normal">Classifier feedback</th>
              <th className="px-4 py-3 text-right font-normal">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#141416]">
            {rows.length > 0 ? (
              rows.map((pred, i) => {
                const key = newsFeedbackKey(pred, i);
                const selected = getRating(key);
                return (
                <tr key={key} className="transition-colors hover:bg-white/[0.015]">
                  <td className={`${expanded ? "max-w-[320px]" : "max-w-[220px]"} px-4 py-3`}>
                    <div className={`${expanded ? "line-clamp-2" : "line-clamp-1"} text-[12px] font-medium text-zinc-200`} title={pred.headline}>
                      {pred.headline}
                    </div>
                  </td>
                  <td className={`${expanded ? "max-w-[360px]" : "max-w-[220px]"} px-4 py-3`}>
                    <div className={`${expanded ? "line-clamp-4" : "line-clamp-1"} text-[11px] leading-relaxed text-zinc-400`} title={pred.rationale}>
                      {pred.rationale}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`rounded px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${severityClass(pred.severity)}`}>
                      {pred.severity}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className={`font-mono text-[11px] uppercase ${
                        pred.direction === 'up' ? 'text-emerald-400'
                          : pred.direction === 'down' ? 'text-rose-400'
                          : 'text-zinc-400'}`}>
                        {pred.direction}
                      </span>
                      {(pred.magnitude_pct_low > 0 || pred.magnitude_pct_high > 0) && (
                        <span className="font-mono text-[10px] text-zinc-500">
                          [{pred.magnitude_pct_low}% – {pred.magnitude_pct_high}%]
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap items-center gap-1.5">
                      {LLM_FEEDBACK_LEVELS.map((level) => {
                        const active = selected === level;
                        return (
                          <button
                            key={level}
                            type="button"
                            onClick={() => sendSeverityFeedback(pred, level, i)}
                            className={`rounded-md border px-2 py-1 text-[10px] font-medium uppercase tracking-wide transition-colors ${
                              active ? `bg-white/10 ${FEEDBACK_TONE[level]}` : `border-[#1f1f22] text-zinc-500 ${FEEDBACK_TONE[level]}`
                            }`}
                            title={`Classify similar future news as ${level.toLowerCase()}`}
                          >
                            {level}
                          </button>
                        );
                      })}
                    </div>
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 text-right font-mono text-[10px] text-zinc-500">
                    {new Date(pred.created_at).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })}
                  </td>
                </tr>
              );
              })
            ) : (
              <tr>
                <td colSpan={6} className="py-10 text-center text-[12px] italic text-zinc-600">
                  No market impacts detected yet (MILD / SIGNIFICANT / SEVERE)
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
