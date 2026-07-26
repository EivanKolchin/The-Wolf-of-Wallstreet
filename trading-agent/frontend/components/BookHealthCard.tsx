"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { Activity, AlertTriangle, Gauge, TrendingUp } from "lucide-react";
import { API_BASE } from "@/lib/api";

interface Health {
  active: boolean;
  marks: number;
  realized_vol: number;
  target_vol: number;
  vol_utilization: number | null;
  realized_sharpe: number;
  current_drawdown: number;
  worst_drawdown: number;
  uptime_coverage: number;
  mark_gaps: number;
  median_mark_minutes: number | null;
  gross: number | null;
  equity: number | null;
  mode: string | null;
  blocked: string | null;
  news_degears: Record<string, string>;
  anomaly_flags: Record<string, number>;
  rebalances: number;
}

const pct = (n: number | null | undefined, d = 1) =>
  n == null ? "—" : `${(n * 100).toFixed(d)}%`;

function Metric({ label, value, tone = "text-zinc-200", sub }: {
  label: string; value: string; tone?: string; sub?: string;
}) {
  return (
    <div className="rounded-lg border border-[#1a1a1c] bg-[#0A0A0A] p-3">
      <div className="text-[10px] uppercase tracking-widest text-zinc-500">{label}</div>
      <div className={`mt-1 font-mono text-lg font-semibold ${tone}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[10px] text-zinc-600">{sub}</div>}
    </div>
  );
}

export function BookHealthCard() {
  const [h, setH] = useState<Health | null>(null);

  useEffect(() => {
    const load = () =>
      fetch(`${API_BASE}/strategy/health`)
        .then((r) => r.json())
        .then(setH)
        .catch(() => {});
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);

  if (!h || !h.active) {
    return (
      <Card>
        <CardHeader className="pb-3 border-b border-[#171717]">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity size={14} className="text-emerald-400" /> Book Health
          </CardTitle>
        </CardHeader>
        <CardContent className="flex h-40 items-center justify-center">
          <p className="text-xs text-zinc-600">
            Waiting for the strategy book to record marks…
          </p>
        </CardContent>
      </Card>
    );
  }

  // vol utilization: 1.0 = at target. Color by how far off.
  const util = h.vol_utilization;
  const utilTone =
    util == null ? "text-zinc-400"
      : util >= 0.8 && util <= 1.25 ? "text-emerald-400"
      : util < 0.5 ? "text-amber-400"
      : "text-zinc-200";
  const sharpeTone = h.realized_sharpe >= 0.5 ? "text-emerald-400"
    : h.realized_sharpe >= 0 ? "text-zinc-200" : "text-rose-400";
  const coverageTone = h.uptime_coverage >= 0.9 ? "text-emerald-400"
    : h.uptime_coverage >= 0.7 ? "text-amber-400" : "text-rose-400";

  const degears = Object.entries(h.news_degears || {});
  const flags = Object.entries(h.anomaly_flags || {});

  return (
    <Card>
      <CardHeader className="pb-3 border-b border-[#171717] flex flex-row items-center justify-between">
        <CardTitle className="text-sm flex items-center gap-2">
          <Activity size={14} className="text-emerald-400" /> Book Health
        </CardTitle>
        <span className="rounded-full border border-[#1f1f22] bg-[#0e0e10] px-2 py-0.5 font-mono text-[10px] text-zinc-500">
          {h.mode || "PAPER"} · {h.marks} marks
        </span>
      </CardHeader>
      <CardContent className="pt-4 space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Metric
            label="Realized Vol"
            value={pct(h.realized_vol)}
            tone={utilTone}
            sub={`target ${pct(h.target_vol)}${util != null ? ` · ${(util * 100).toFixed(0)}% of target` : ""}`}
          />
          <Metric
            label="Realized Sharpe"
            value={h.realized_sharpe.toFixed(2)}
            tone={sharpeTone}
            sub="annualized, live marks"
          />
          <Metric
            label="Drawdown"
            value={pct(h.current_drawdown)}
            tone={h.current_drawdown < -0.1 ? "text-rose-400" : "text-zinc-200"}
            sub={`worst ${pct(h.worst_drawdown)}`}
          />
          <Metric
            label="Uptime"
            value={pct(h.uptime_coverage, 0)}
            tone={coverageTone}
            sub={`${h.mark_gaps} gap(s)${h.median_mark_minutes ? ` · ~${h.median_mark_minutes}m cadence` : ""}`}
          />
        </div>

        {util != null && util < 0.5 && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 p-2.5 text-[11px] text-amber-300">
            <Gauge size={13} className="mt-0.5 shrink-0" />
            Book is running at {(util * 100).toFixed(0)}% of its {pct(h.target_vol)} vol target — under-risked vs intent.
          </div>
        )}
        {h.blocked && (
          <div className="flex items-center gap-2 rounded-lg border border-rose-500/20 bg-rose-500/5 p-2.5 text-[11px] text-rose-300">
            <AlertTriangle size={13} /> Rebalance blocked: {h.blocked}
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div>
            <div className="mb-1.5 text-[10px] uppercase tracking-widest text-zinc-500">
              Active news de-gears
            </div>
            {degears.length === 0 ? (
              <p className="text-[11px] text-zinc-600">None — no news tightening exposure.</p>
            ) : (
              <div className="space-y-1">
                {degears.slice(0, 5).map(([sym, note]) => (
                  <div key={sym} className="flex items-center gap-2 text-[11px]">
                    <span className="font-mono text-zinc-300">{sym.replace("USDT", "").replace("-USD", "")}</span>
                    <span className="text-zinc-500">{note}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <div className="mb-1.5 text-[10px] uppercase tracking-widest text-zinc-500 flex items-center gap-1">
              <TrendingUp size={11} /> Anomaly flags
            </div>
            {flags.length === 0 ? (
              <p className="text-[11px] text-zinc-600">Universe clean — no wide-spread / illiquid / abnormal flags.</p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {flags.slice(0, 10).map(([sym, n]) => (
                  <span key={sym} className="rounded-md border border-amber-500/20 bg-amber-500/5 px-1.5 py-0.5 font-mono text-[10px] text-amber-300">
                    {sym.replace("USDT", "")} ·{n}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
