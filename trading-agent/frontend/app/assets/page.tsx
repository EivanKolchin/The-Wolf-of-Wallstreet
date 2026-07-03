"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { API_BASE } from "@/lib/api";

interface ExtendedQuote {
  price: number; session: string; is_extended: boolean; source: string;
}
interface AssetRow {
  symbol: string;
  asset_class: string;
  session: string;
  last_price: number | null;
  volume: number;
  volatility_pct: number | null;
  price_change_pct: number | null;
  spark: number[];
  extended_hours: ExtendedQuote | null;
  position: any | null;
}

const TIMEFRAMES = {
  day: { label: "Last Day", interval: "15m", limit: 96 },
  week: { label: "Last Week", interval: "1h", limit: 24 * 7 },
  month: { label: "Last Month", interval: "4h", limit: 30 * 6 },
  year: { label: "Last Year", interval: "1d", limit: 365 },
} as const;

const SORTS = [
  { id: "top", label: "Top Trend" },
  { id: "bottom", label: "Bottom Trend" },
  { id: "uptrend", label: "Strongest Uptrend" },
  { id: "downtrend", label: "Strongest Downtrend" },
  { id: "volatility", label: "Most Volatile" },
  { id: "volume", label: "Highest Volume" },
] as const;

const VIEW_MODES = [
  { id: "flat", label: "Flat Ranking" },
  { id: "assetType", label: "By Asset Type" },
] as const;

const STOCK_THEMES = {
  memory: "Memory",
  semiconductors: "Semiconductors",
  ai: "AI / Software",
  healthcare: "Healthcare",
  space: "Space",
  crypto_fintech: "Crypto / Fintech",
  ev_auto: "EV / Auto",
  energy: "Energy",
  other: "Other",
} as const;

const STOCK_THEME_BY_SYMBOL: Record<string, keyof typeof STOCK_THEMES> = {
  SNDK: "memory",
  MU: "memory",
  AMD: "semiconductors",
  NVDA: "semiconductors",
  TSM: "semiconductors",
  SMCI: "semiconductors",
  PLTR: "ai",
  COIN: "crypto_fintech",
  MSTR: "crypto_fintech",
  TSLA: "ev_auto",
  BE: "energy",
};

const displaySymbol = (s: string) => s.replace("USDT", "");

function fmtNum(n: number | null | undefined, d = 2) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}
function fmtCompact(n: number | null | undefined) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return n.toFixed(2);
}

function sessionBadge(session: string) {
  const map: Record<string, string> = {
    regular: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    open: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    pre: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    after: "bg-orange-500/15 text-orange-400 border-orange-500/30",
    overnight: "bg-sky-500/15 text-sky-400 border-sky-500/30",
    closed: "bg-zinc-700/30 text-zinc-400 border-zinc-700/50",
  };
  const label: Record<string, string> = {
    regular: "Regular", open: "24/7", pre: "Pre-Market", after: "After-Hours",
    overnight: "Overnight", closed: "Closed",
  };
  return { cls: map[session] || map.closed, label: label[session] || session };
}

function Sparkline({ data, up }: { data: number[]; up: boolean }) {
  if (!data || data.length < 2) {
    return <div className="flex h-12 items-center justify-center text-[10px] text-zinc-600">no data</div>;
  }
  const w = 240, h = 48, pad = 2;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / range) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-12 w-full" preserveAspectRatio="none">
      <polyline
        points={pts}
        fill="none"
        stroke={up ? "#34d399" : "#f87171"}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function AssetCard({ a, rank, timeframeLabel }: { a: AssetRow; rank: number; timeframeLabel: string }) {
  const up = (a.price_change_pct ?? 0) >= 0;
  const sb = sessionBadge(a.session);
  return (
    <Card className="flex flex-col gap-3 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="rounded-md border border-zinc-700/60 px-1.5 py-0.5 text-[9px] font-semibold text-zinc-400">
            #{rank}
          </span>
          <span className="text-base font-semibold text-zinc-100">{displaySymbol(a.symbol)}</span>
          <span className="rounded border border-zinc-700/60 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-zinc-400">
            {a.asset_class === "crypto" ? "Crypto" : "Stock"}
          </span>
        </div>
        <span className={`rounded border px-1.5 py-0.5 text-[9px] uppercase tracking-wider ${sb.cls}`}>{sb.label}</span>
      </div>

      <div className="flex items-end justify-between">
        <div className="text-xl font-mono font-semibold text-zinc-100">
          ${fmtNum(a.last_price, a.last_price && a.last_price < 10 ? 4 : 2)}
        </div>
        <div className={`text-sm font-mono ${up ? "text-emerald-400" : "text-rose-400"}`}>
          {up ? "+" : ""}{fmtNum(a.price_change_pct, 2)}%
        </div>
      </div>

      <Sparkline data={a.spark} up={up} />

      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div className="rounded border border-zinc-800/50 bg-[#0A0A0A] p-2">
          <div className="text-[9px] uppercase tracking-wider text-zinc-500">Volatility</div>
          <div className="font-mono text-zinc-200">{fmtNum(a.volatility_pct, 2)}%</div>
        </div>
        <div className="rounded border border-zinc-800/50 bg-[#0A0A0A] p-2">
          <div className="text-[9px] uppercase tracking-wider text-zinc-500">Volume</div>
          <div className="font-mono text-zinc-200">{fmtCompact(a.volume)}</div>
        </div>
        <div className="rounded border border-zinc-800/50 bg-[#0A0A0A] p-2">
          <div className="text-[9px] uppercase tracking-wider text-zinc-500">{timeframeLabel}</div>
          <div className={`font-mono ${up ? "text-emerald-400" : "text-rose-400"}`}>{up ? "+" : ""}{fmtNum(a.price_change_pct, 1)}%</div>
        </div>
      </div>

      {a.extended_hours && (
        <div className="flex items-center justify-between rounded border border-amber-500/20 bg-amber-500/5 px-2 py-1 text-[10px]">
          <span className="uppercase tracking-wider text-amber-400">{a.extended_hours.session} px</span>
          <span className="font-mono text-zinc-200">${fmtNum(a.extended_hours.price, 2)}</span>
          <span className="text-zinc-600">via {a.extended_hours.source}</span>
        </div>
      )}

      <div className="border-t border-zinc-800/50 pt-2">
        {a.position ? (
          <div className="flex items-center justify-between text-[11px]">
            <span className={`font-bold uppercase ${a.position.direction === "short" ? "text-rose-400" : "text-emerald-400"}`}>
              {(a.position.direction || "long").toUpperCase()} ${fmtNum(a.position.size_usd, 0)}
            </span>
            <span className={`font-mono ${(a.position.unrealized ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
              {(a.position.unrealized ?? 0) >= 0 ? "+" : ""}{fmtNum(a.position.unrealized, 2)} USD
            </span>
          </div>
        ) : (
          <div className="text-[10px] text-zinc-600">No open position</div>
        )}
      </div>
    </Card>
  );
}

function sortAssets(rows: AssetRow[], sortId: string) {
  const byMetric = (selector: (a: AssetRow) => number) => [...rows].sort((a, b) => selector(b) - selector(a));
  switch (sortId) {
    case "bottom":
    case "downtrend":
      return [...rows].sort((a, b) => (a.price_change_pct ?? 0) - (b.price_change_pct ?? 0));
    case "volatility":
      return byMetric(a => a.volatility_pct ?? -Infinity);
    case "volume":
      return byMetric(a => a.volume ?? -Infinity);
    case "uptrend":
    case "top":
    default:
      return byMetric(a => a.price_change_pct ?? -Infinity);
  }
}

function stockThemeFor(symbol: string) {
  const upper = symbol.toUpperCase();
  return STOCK_THEME_BY_SYMBOL[upper] || "other";
}

function groupAssetsByType(rows: AssetRow[]) {
  const crypto: AssetRow[] = [];
  const stockThemes = new Map<keyof typeof STOCK_THEMES, AssetRow[]>();

  for (const row of rows) {
    if (row.asset_class === "crypto") {
      crypto.push(row);
      continue;
    }
    const theme = stockThemeFor(row.symbol);
    const current = stockThemes.get(theme) || [];
    current.push(row);
    stockThemes.set(theme, current);
  }

  return { crypto, stockThemes };
}

export default function AllAssetsPage() {
  const [assets, setAssets] = React.useState<AssetRow[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [err, setErr] = React.useState<string | null>(null);
  const [timeframe, setTimeframe] = React.useState<keyof typeof TIMEFRAMES>("day");
  const [search, setSearch] = React.useState("");
  const [assetFilter, setAssetFilter] = React.useState<"all" | "crypto" | "stocks">("all");
  const [sortBy, setSortBy] = React.useState<(typeof SORTS)[number]["id"]>("top");
  const [viewMode, setViewMode] = React.useState<(typeof VIEW_MODES)[number]["id"]>("flat");

  React.useEffect(() => {
    let cancelled = false;
    const tf = TIMEFRAMES[timeframe];
    const load = () => {
      const qs = new URLSearchParams({
        interval: tf.interval,
        limit: String(tf.limit),
      });
      fetch(`${API_BASE}/assets/overview?${qs.toString()}`)
        .then(r => r.json())
        .then(d => {
          if (cancelled) return;
          setAssets(Array.isArray(d?.assets) ? d.assets : []);
          setErr(null);
        })
        .catch(e => { if (!cancelled) setErr(String(e)); })
        .finally(() => { if (!cancelled) setLoading(false); });
    };
    setLoading(true);
    load();
    const id = setInterval(load, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, [timeframe]);

  const filteredAssets = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    return assets.filter((a) => {
      if (assetFilter === "crypto" && a.asset_class !== "crypto") return false;
      if (assetFilter === "stocks" && a.asset_class === "crypto") return false;
      if (!q) return true;
      return a.symbol.toLowerCase().includes(q) || displaySymbol(a.symbol).toLowerCase().includes(q);
    });
  }, [assets, assetFilter, search]);

  const rankedAssets = React.useMemo(() => sortAssets(filteredAssets, sortBy), [filteredAssets, sortBy]);
  const groupedAssets = React.useMemo(() => groupAssetsByType(rankedAssets), [rankedAssets]);
  const timeframeLabel = TIMEFRAMES[timeframe].label.replace("Last ", "");

  const renderAssetCard = (a: AssetRow, idx: number) => (
    <AssetCard key={`${a.symbol}-${timeframe}`} a={a} rank={idx + 1} timeframeLabel={timeframeLabel} />
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white">All Assets</h1>
          <p className="text-sm text-zinc-500">Live cross-asset overview with search, ranking, and timeframe-based trend filters.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {Object.entries(TIMEFRAMES).map(([id, tf]) => (
            <button
              key={id}
              onClick={() => setTimeframe(id as keyof typeof TIMEFRAMES)}
              className={`rounded-md px-3 py-1.5 text-[12px] font-medium transition-colors ${
                timeframe === id ? "bg-zinc-100 text-zinc-900" : "border border-[#1a1a1c] bg-[#0e0e10] text-zinc-400 hover:text-zinc-100"
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      <Card className="p-4">
        <div className="grid gap-3 lg:grid-cols-[1.4fr_0.8fr_1fr]">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets by ticker..."
            className="rounded-lg border border-[#1f1f22] bg-[#0e0e10] px-3 py-2 text-sm text-zinc-200 outline-none placeholder:text-zinc-600"
          />
          <div className="flex gap-2">
            {(["all", "crypto", "stocks"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setAssetFilter(mode)}
                className={`flex-1 rounded-md px-3 py-2 text-[12px] font-medium capitalize transition-colors ${
                  assetFilter === mode ? "bg-zinc-100 text-zinc-900" : "border border-[#1a1a1c] bg-[#0e0e10] text-zinc-400 hover:text-zinc-100"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            {VIEW_MODES.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setViewMode(mode.id)}
                className={`flex-1 rounded-md px-3 py-2 text-[12px] font-medium transition-colors ${
                  viewMode === mode.id ? "bg-zinc-100 text-zinc-900" : "border border-[#1a1a1c] bg-[#0e0e10] text-zinc-400 hover:text-zinc-100"
                }`}
              >
                {mode.label}
              </button>
            ))}
          </div>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as (typeof SORTS)[number]["id"])}
            className="rounded-lg border border-[#1f1f22] bg-[#0e0e10] px-3 py-2 text-sm text-zinc-200 outline-none"
          >
            {SORTS.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        </div>
      </Card>

      {err && <div className="text-sm text-rose-400">Failed to load: {err}</div>}
      {loading && <div className="text-xs text-zinc-500">loading…</div>}

      {viewMode === "flat" ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {rankedAssets.map((a, idx) => renderAssetCard(a, idx))}
        </div>
      ) : (
        <div className="space-y-8">
          {groupedAssets.crypto.length > 0 && (
            <section className="space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold tracking-tight text-white">Crypto</h2>
                <span className="text-xs text-zinc-500">{groupedAssets.crypto.length} assets</span>
              </div>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {groupedAssets.crypto.map((a, idx) => renderAssetCard(a, idx))}
              </div>
            </section>
          )}

          {Array.from(groupedAssets.stockThemes.entries()).map(([theme, rows]) => (
            rows.length > 0 ? (
              <section key={theme} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold tracking-tight text-white">{STOCK_THEMES[theme]}</h2>
                  <span className="text-xs text-zinc-500">{rows.length} assets</span>
                </div>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                  {rows.map((a, idx) => renderAssetCard(a, idx))}
                </div>
              </section>
            ) : null
          ))}

          {groupedAssets.crypto.length === 0 && groupedAssets.stockThemes.size === 0 && !loading && !err && (
            <div className="text-sm text-zinc-500">No assets match the current search/filter.</div>
          )}
        </div>
      )}

      {!loading && rankedAssets.length === 0 && !err && viewMode === "flat" && (
        <div className="text-sm text-zinc-500">No assets match the current search/filter.</div>
      )}
    </div>
  );
}
