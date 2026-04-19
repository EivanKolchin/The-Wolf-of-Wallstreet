"use client";
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { BrainCircuit } from "lucide-react";

export function NewsInsightsWidget({ predictions }: { predictions: any[] }) {
  return (
    <Card className="h-full flex flex-col overflow-hidden">
      <CardHeader className="border-b border-zinc-800/50 pb-4 flex flex-row items-center justify-between">
        <CardTitle className="text-sm">LLM Intel Log</CardTitle>
      </CardHeader>
      <CardContent className="font-sans text-[12px] text-zinc-400 p-0 flex-1 overflow-y-auto max-h-[300px]">
        <table className="w-full text-left">
          <thead className="sticky top-0 bg-[#161616] border-b border-zinc-800/50 text-[10px] uppercase text-zinc-500 font-medium">
            <tr>
              <th className="py-3 px-4 font-normal">Summary</th>
              <th className="py-3 px-4 font-normal">Rating</th>
              <th className="py-3 px-4 font-normal">Prediction</th>
              <th className="py-3 px-4 text-right font-normal">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/30">
            {predictions && predictions.length > 0 ? (
              predictions.slice(0, 10).map((pred, i) => (
                <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3 px-4 max-w-[200px]">
                    <div className="text-[#D1D4DC] font-medium line-clamp-1 mb-0.5" title={pred.headline}>{pred.headline}</div>
                    <div className="text-zinc-500 text-[10px] line-clamp-1" title={pred.rationale}>{pred.rationale}</div>
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded text-[9px] uppercase tracking-wider font-semibold 
                      ${pred.severity === 'SEVERE' ? 'bg-rose-500/10 text-rose-400' : 
                        pred.severity === 'SIGNIFICANT' ? 'bg-amber-500/10 text-amber-400' : 
                        'bg-zinc-800 text-zinc-400'}`}
                    >
                      {pred.severity}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                       <span className={`font-mono text-[11px] uppercase
                        ${pred.direction === 'up' ? 'text-emerald-400' : 
                          pred.direction === 'down' ? 'text-rose-400' : 
                          'text-zinc-400'}`}
                        >
                          {pred.direction}
                        </span>
                        {(pred.magnitude_pct_low > 0 || pred.magnitude_pct_high > 0) && (
                           <span className="text-[10px] text-zinc-500 font-mono">
                             [{pred.magnitude_pct_low}% - {pred.magnitude_pct_high}%]
                           </span>
                        )}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-[10px] text-zinc-500 whitespace-nowrap">
                    {new Date(pred.created_at).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute:'2-digit' })}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} className="py-8 text-center text-zinc-600 italic">
                  No critical impacts detected yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}