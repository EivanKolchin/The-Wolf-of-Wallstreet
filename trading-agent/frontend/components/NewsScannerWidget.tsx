"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { newsFeedbackKey, useNewsFeedback } from "@/lib/hooks/useNewsFeedback";

export function NewsScannerWidget({ rawNews, className = "" }: { rawNews: any[]; className?: string }) {
  const count = rawNews?.length || 0;
  const { getRating, submitFeedback } = useNewsFeedback();

  const sendFeedback = async (news: any, rating: 1 | -1, index: number) => {
    const key = newsFeedbackKey(news, index);
    await submitFeedback(key, rating, {
      feedback_type: "relevance",
      headline: news.headline,
      source_domain: news.source,
      article_hash: news.article_hash,
    });
  };

  return (
    <Card className={`flex flex-col overflow-hidden ${className}`}>
      <CardHeader className="flex flex-row items-center justify-between border-b border-[#171717] pb-4">
        <CardTitle className="text-sm">Live Agent Scanner</CardTitle>
        <span className="rounded-full border border-[#1f1f22] bg-[#0e0e10] px-2 py-0.5 font-mono text-[10px] text-zinc-500">
          {count ? `${count} events` : "idle"}
        </span>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 space-y-2.5 overflow-y-auto pt-4">
        {count > 0 ? (
          rawNews.map((news, i) => {
            const key = newsFeedbackKey(news, i);
            const selected = getRating(key);
            return (
            <div key={key} className="border-b border-[#141416] pb-2.5 last:border-0">
              <div className="mb-1 flex items-center justify-between text-[10px] text-zinc-600">
                <span className="uppercase tracking-wider">{news.source}</span>
                <span className="font-mono">{new Date(news.time).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit' })}</span>
              </div>
              <p className="line-clamp-2 text-[12px] leading-relaxed text-zinc-300" title={news.headline}>
                {news.headline}
              </p>
              <div className="mt-2 flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => sendFeedback(news, 1, i)}
                  className={`rounded-md border px-2 py-1 transition-colors ${selected === 1 ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300" : "border-[#1f1f22] text-zinc-500 hover:border-emerald-500/40 hover:text-emerald-300"}`}
                  title="Scan and analyse more news like this"
                >
                  <ThumbsUp size={12} />
                </button>
                <button
                  type="button"
                  onClick={() => sendFeedback(news, -1, i)}
                  className={`rounded-md border px-2 py-1 transition-colors ${selected === -1 ? "border-rose-500/50 bg-rose-500/10 text-rose-300" : "border-[#1f1f22] text-zinc-500 hover:border-rose-500/40 hover:text-rose-300"}`}
                  title="Deprioritize news like this"
                >
                  <ThumbsDown size={12} />
                </button>
              </div>
            </div>
          );
          })
        ) : (
          <div className="flex h-full items-center justify-center py-10 text-[12px] italic text-zinc-600">
            Listening to global feeds…
          </div>
        )}
      </CardContent>
    </Card>
  );
}
