"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Radar } from "lucide-react";

export function NewsScannerWidget({ rawNews }: { rawNews: any[] }) {
  return (
    <Card className="h-full flex flex-col overflow-hidden">
      <CardHeader className="border-b border-zinc-800/50 pb-4 flex flex-row items-center justify-between">
        <CardTitle className="text-sm">Live Agent Scanner</CardTitle>
        <Radar size={15} className="text-emerald-500 animate-pulse" />
      </CardHeader>
      <CardContent className="font-mono text-[11px] text-zinc-400 space-y-3 pt-4 flex-1 overflow-y-auto max-h-[300px]">
        {rawNews && rawNews.length > 0 ? (
          rawNews.map((news, i) => (
            <div key={i} className="flex flex-col gap-1 border-b border-zinc-800/30 pb-2">
              <div className="flex justify-between items-center text-zinc-500 text-[10px]">
                <span className="uppercase tracking-wider">{news.source}</span>
                <span>{new Date(news.time).toLocaleTimeString([], { hour12: false, second: '2-digit' })}</span>
              </div>
              <p className="text-[#D1D4DC] leading-relaxed line-clamp-2" title={news.headline}>
                {news.headline}
              </p>
            </div>
          ))
        ) : (
          <div className="flex items-center justify-center h-full text-zinc-600 italic">
            Listening to global feeds...
          </div>
        )}
      </CardContent>
    </Card>
  );
}