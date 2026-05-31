"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import { RefreshCw, RotateCcw, AlertCircle, Activity, Newspaper, Cpu, Clock, Target } from "lucide-react";

type Stats = any;

const TABS = ["Overall", "Today", "Per-Asset", "Model & News", "Latency"] as const;
type Tab = typeof TABS[number];

export default function PerformancePage() {
  const [data, setData] = useState<Stats | null>(null);
  const [tab, setTab] = useState<Tab>("Overall");
  const [loading, setLoading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<number | null>(null);

  const fetchStats = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/stats/performance`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = await res.json();
      setData(j);
      setError(null);
      setLastFetched(Date.now());
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  };

  const doReset = async (clear: boolean) => {
    setResetting(true);
    try {
      await fetch(`${API_BASE}/stats/reset`, { method: clear ? "DELETE" : "POST" });
      await fetchStats();
    } catch {}
    setResetting(false);
    setConfirmReset(false);
  };

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="space-y-6 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Model Performance</h1>
          <p className="text-zinc-500 text-xs mt-1">
            Aggregates trades, news classification, and outbound API latency. Polls every 5s.
            {data?.reset_at && (
              <span className="ml-2 text-amber-400">
                Baseline reset {new Date(data.reset_at * 1000).toLocaleString()}
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={fetchStats}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-1.5 text-xs rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition-colors"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} /> Refresh
          </button>
          {confirmReset ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-amber-400">Are you sure?</span>
              <button
                onClick={() => doReset(false)}
                disabled={resetting}
                className="flex items-center gap-2 px-3 py-1.5 text-xs rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 transition-colors"
              >
                <RotateCcw size={12} /> Confirm Reset
              </button>
              {data?.reset_at && (
                <button
                  onClick={() => doReset(true)}
                  className="px-3 py-1.5 text-xs rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors"
                >
                  Clear baseline (count from genesis)
                </button>
              )}
              <button
                onClick={() => setConfirmReset(false)}
                className="px-3 py-1.5 text-xs rounded text-zinc-400 hover:text-zinc-200 transition-colors"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmReset(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-xs rounded bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 text-amber-400 transition-colors"
            >
              <RotateCcw size={12} /> Reset Stats Baseline
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="flex items-center gap-2 px-4 py-2 rounded bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* Runtime markers */}
      {data?.meta && <RuntimeBanner meta={data.meta} lastFetched={lastFetched} />}

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b border-zinc-800">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-xs transition-colors ${
              tab === t ? "text-white border-b-2 border-violet-400" : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {!data ? (
        <div className="text-zinc-500 text-sm">Loading…</div>
      ) : (
        <>
          {tab === "Overall" && <TradeStatsGrid title="All time (since baseline)" s={data.overall} />}
          {tab === "Today" && <TradeStatsGrid title="Today (UTC)" s={data.today} />}
          {tab === "Per-Asset" && <PerSymbolView per={data.per_symbol} />}
          {tab === "Model & News" && <ModelAndNews meta={data.meta} news={data.news} />}
          {tab === "Latency" && <LatencyView lat={data.latency} />}
        </>
      )}
    </div>
  );
}

function RuntimeBanner({ meta, lastFetched }: { meta: any; lastFetched: number | null }) {
  const uptime = meta.agent_uptime_seconds || 0;
  const hrs = Math.floor(uptime / 3600);
  const mins = Math.floor((uptime % 3600) / 60);
  const buf = meta.buffer_current ?? 0;
  const need = meta.buffer_required ?? 0;
  const bufPct = need ? Math.min(100, Math.round((buf / need) * 100)) : 0;
  return (
    <div className="grid grid-cols-2 md:grid-cols-6 gap-3 text-xs">
      <Stat icon={<Cpu size={12} />} label="Feature ver." value={meta.feature_version ?? "?"} />
      <Stat icon={<Clock size={12} />} label="Agent uptime" value={`${hrs}h ${mins}m`} />
      <Stat label="Buffer fill" value={`${buf} / ${need} (${bufPct}%)`} />
      <Stat label="Cycle interval" value={`${meta.cycle_interval ?? "?"}s`} />
      <Stat label="Market data" value={meta.has_market_data ? "yes" : "no"} tone={meta.has_market_data ? "ok" : "warn"} />
      <Stat label="Halted" value={meta.is_halted ? "YES" : "no"} tone={meta.is_halted ? "bad" : "ok"} />
    </div>
  );
}

function Stat({ label, value, icon, tone = "neutral" }: { label: string; value: any; icon?: any; tone?: "ok"|"bad"|"warn"|"neutral" }) {
  const toneClass =
    tone === "ok" ? "border-emerald-500/20 text-emerald-400" :
    tone === "bad" ? "border-red-500/20 text-red-400" :
    tone === "warn" ? "border-amber-500/20 text-amber-400" :
    "border-zinc-800 text-zinc-200";
  return (
    <div className={`flex flex-col px-3 py-2 rounded border bg-zinc-900/50 ${toneClass}`}>
      <span className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-zinc-500">{icon}{label}</span>
      <span className="font-mono mt-0.5 text-sm">{value}</span>
    </div>
  );
}

function TradeStatsGrid({ title, s }: { title: string; s: any }) {
  if (!s) return null;
  const pnlPos = (s.cumulative_pnl_usd ?? 0) >= 0;
  return (
    <section className="space-y-4">
      <h2 className="text-sm uppercase tracking-wider text-zinc-400">{title}</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <BigCard title="Cumulative PnL" value={`$${(s.cumulative_pnl_usd || 0).toFixed(2)}`} sub={`${(s.cumulative_pnl_pct || 0).toFixed(2)}%`} tone={pnlPos ? "ok" : "bad"} />
        <BigCard title="Win rate" value={`${((s.win_rate || 0) * 100).toFixed(1)}%`} sub={`${s.wins} W / ${s.losses} L`} />
        <BigCard title="Expectancy / trade" value={`${(s.expectancy_pct || 0).toFixed(3)}%`} />
        <BigCard title="Profit factor" value={s.profit_factor === null ? "∞" : (s.profit_factor || 0).toFixed(2)} />

        <BigCard title="Trades total" value={`${s.total_trades}`} sub={`${s.open_trades} open / ${s.closed_trades} closed`} />
        <BigCard title="Trades / day" value={(s.trades_per_day || 0).toFixed(2)} />
        <BigCard title="Avg holding" value={`${(s.avg_holding_minutes || 0).toFixed(1)} min`} />
        <BigCard title="Avg size" value={`$${(s.avg_size_usd || 0).toFixed(2)}`} />

        <BigCard title="Sharpe (per-trade)" value={(s.sharpe_per_trade || 0).toFixed(2)} />
        <BigCard title="Sortino (per-trade)" value={(s.sortino_per_trade || 0).toFixed(2)} />
        <BigCard title="Best trade" value={`+${(s.best_trade_pct || 0).toFixed(2)}%`} tone="ok" />
        <BigCard title="Worst trade" value={`${(s.worst_trade_pct || 0).toFixed(2)}%`} tone="bad" />

        <BigCard title="Avg NN confidence" value={`${((s.avg_nn_confidence || 0) * 100).toFixed(1)}%`} />
        <BigCard title="Long / Short" value={`${s.long_trades} / ${s.short_trades}`} />
        <BigCard title="Streak (W / L)" value={`${s.longest_win_streak} / ${s.longest_loss_streak}`} />
        <BigCard title="Avg target-price error" value={`${(s.avg_abs_target_err_pct || 0).toFixed(3)}%`} />
      </div>

      {s.exit_reasons && Object.keys(s.exit_reasons).length > 0 && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4">
          <h3 className="text-xs uppercase tracking-wider text-zinc-400 mb-2">Exit reasons</h3>
          <div className="flex flex-wrap gap-3">
            {Object.entries(s.exit_reasons as Record<string, number>).map(([k, n]) => (
              <span key={k} className="px-3 py-1 rounded bg-zinc-800/60 text-xs text-zinc-200">
                {k}: <span className="font-mono text-violet-400">{n}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function BigCard({ title, value, sub, tone = "neutral" }: { title: string; value: string; sub?: string; tone?: "ok"|"bad"|"neutral" }) {
  const valueClass =
    tone === "ok" ? "text-emerald-400" :
    tone === "bad" ? "text-red-400" :
    "text-white";
  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-3">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{title}</div>
      <div className={`text-lg font-semibold font-mono mt-0.5 ${valueClass}`}>{value}</div>
      {sub && <div className="text-[11px] text-zinc-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function PerSymbolView({ per }: { per: Record<string, any> }) {
  const syms = Object.keys(per || {});
  if (!syms.length) return <div className="text-zinc-500 text-sm">No per-symbol stats yet.</div>;
  return (
    <section className="space-y-3">
      <h2 className="text-sm uppercase tracking-wider text-zinc-400">By asset</h2>
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs whitespace-nowrap">
            <thead className="uppercase tracking-wider border-b border-zinc-800 bg-zinc-950/50 text-zinc-400">
              <tr>
                <th className="px-4 py-3">Asset</th>
                <th className="px-4 py-3">Total</th>
                <th className="px-4 py-3">Open</th>
                <th className="px-4 py-3">Win rate</th>
                <th className="px-4 py-3">PnL ($)</th>
                <th className="px-4 py-3">PnL (%)</th>
                <th className="px-4 py-3">Sharpe</th>
                <th className="px-4 py-3">Expectancy</th>
                <th className="px-4 py-3">Trades/day</th>
                <th className="px-4 py-3">Avg hold</th>
                <th className="px-4 py-3">Best</th>
                <th className="px-4 py-3">Worst</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50">
              {syms.map((sym) => {
                const s = per[sym];
                const pos = (s.cumulative_pnl_usd || 0) >= 0;
                return (
                  <tr key={sym} className="hover:bg-zinc-800/20">
                    <td className="px-4 py-3 font-medium text-white">{sym}</td>
                    <td className="px-4 py-3 text-zinc-300">{s.total_trades}</td>
                    <td className="px-4 py-3 text-zinc-300">{s.open_trades}</td>
                    <td className="px-4 py-3 text-zinc-300">{((s.win_rate || 0) * 100).toFixed(1)}%</td>
                    <td className={`px-4 py-3 font-mono ${pos ? "text-emerald-400" : "text-red-400"}`}>
                      {(s.cumulative_pnl_usd || 0).toFixed(2)}
                    </td>
                    <td className={`px-4 py-3 font-mono ${pos ? "text-emerald-400" : "text-red-400"}`}>
                      {(s.cumulative_pnl_pct || 0).toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-zinc-300">{(s.sharpe_per_trade || 0).toFixed(2)}</td>
                    <td className="px-4 py-3 text-zinc-300">{(s.expectancy_pct || 0).toFixed(3)}%</td>
                    <td className="px-4 py-3 text-zinc-300">{(s.trades_per_day || 0).toFixed(2)}</td>
                    <td className="px-4 py-3 text-zinc-300">{(s.avg_holding_minutes || 0).toFixed(1)}m</td>
                    <td className="px-4 py-3 text-emerald-400">{(s.best_trade_pct || 0).toFixed(2)}%</td>
                    <td className="px-4 py-3 text-red-400">{(s.worst_trade_pct || 0).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function ModelAndNews({ meta, news }: { meta: any; news: any }) {
  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-sm uppercase tracking-wider text-zinc-400 mb-3 flex items-center gap-2">
          <Newspaper size={14} /> News pipeline
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <BigCard title="News processed" value={`${news.total || 0}`} />
          <BigCard title="Minor / neutral" value={`${news.minor_count || 0}`} />
          <BigCard title="Significant" value={`${news.significant_count || 0}`} tone="ok" />
          <BigCard title="Severe" value={`${news.severe_count || 0}`} tone="bad" />
          <BigCard title="Avg confidence" value={`${((news.avg_confidence || 0) * 100).toFixed(1)}%`} />
          <BigCard title="Outcome checked" value={`${news.outcome_checked || 0}`} />
          <BigCard title="Avg prediction score" value={(news.avg_prediction_score || 0).toFixed(3)} />
          <BigCard title="Avg actual move" value={`${(news.avg_actual_move_pct || 0).toFixed(2)}%`} />
        </div>
      </div>

      <div>
        <h2 className="text-sm uppercase tracking-wider text-zinc-400 mb-3 flex items-center gap-2">
          <Activity size={14} /> Live attention
        </h2>
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4">
          {meta?.attention_state && Object.keys(meta.attention_state).length ? (
            <div className="flex flex-wrap gap-3 text-xs">
              {Object.entries(meta.attention_state as Record<string, string>).map(([sym, lvl]) => (
                <span key={sym} className={`px-3 py-1 rounded font-mono ${lvl === "high" ? "bg-red-500/20 text-red-300" : "bg-emerald-500/20 text-emerald-300"}`}>
                  {sym}: {lvl}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-xs text-zinc-500">No attention state in Redis yet.</span>
          )}
        </div>
      </div>
    </section>
  );
}

function LatencyView({ lat }: { lat: any }) {
  const totals = lat?.totals || {};
  const provs = lat?.providers || {};
  return (
    <section className="space-y-4">
      <h2 className="text-sm uppercase tracking-wider text-zinc-400 flex items-center gap-2">
        <Target size={14} /> Outbound API latency
      </h2>
      <div className="grid grid-cols-3 gap-3">
        <BigCard title="Total calls" value={`${totals.calls || 0}`} />
        <BigCard title="Errors" value={`${totals.errors || 0}`} tone={totals.errors ? "bad" : "ok"} />
        <BigCard title="Error rate" value={`${((totals.error_rate || 0) * 100).toFixed(2)}%`} tone={(totals.error_rate || 0) > 0.05 ? "bad" : "ok"} />
      </div>
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-left text-xs whitespace-nowrap">
          <thead className="uppercase tracking-wider border-b border-zinc-800 bg-zinc-950/50 text-zinc-400">
            <tr>
              <th className="px-4 py-3">Provider</th>
              <th className="px-4 py-3">Calls</th>
              <th className="px-4 py-3">Errors</th>
              <th className="px-4 py-3">Error rate</th>
              <th className="px-4 py-3">Avg latency</th>
              <th className="px-4 py-3">p50</th>
              <th className="px-4 py-3">p95</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {Object.keys(provs).length === 0 && (
              <tr><td colSpan={7} className="px-4 py-8 text-center text-zinc-500">No API calls logged yet.</td></tr>
            )}
            {Object.entries(provs).map(([prov, p]: any) => (
              <tr key={prov}>
                <td className="px-4 py-3 font-mono text-zinc-200">{prov}</td>
                <td className="px-4 py-3 text-zinc-300">{p.calls}</td>
                <td className="px-4 py-3 text-zinc-300">{p.errors}</td>
                <td className={`px-4 py-3 ${p.error_rate > 0.05 ? "text-red-400" : "text-zinc-300"}`}>{(p.error_rate * 100).toFixed(2)}%</td>
                <td className="px-4 py-3 text-zinc-300">{p.avg_latency_ms} ms</td>
                <td className="px-4 py-3 text-zinc-300">{p.p50_latency_ms} ms</td>
                <td className="px-4 py-3 text-zinc-300">{p.p95_latency_ms} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
