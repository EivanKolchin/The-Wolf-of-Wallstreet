"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { API_BASE } from "@/lib/api";

// ---- types -----------------------------------------------------------------
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

const STOCKS = new Set(["SNDK", "AMD", "MU", "AXTI", "BE"]);
const isStock = (s: string) => STOCKS.has((s || "").toUpperCase());
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

// Color-code the market session badge.
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

// Lightweight inline SVG sparkline — avoids spinning up N chart instances.
function Sparkline({ data, up }: { data: number[]; up: boolean }) {
  if (!data || data.length < 2) {
    return <div className="h-12 flex items-center justify-center text-[10px] text-zinc-600">no data</div>;
  }
  const w = 240, h = 48, pad = 2;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
    const y = pad + (1 - (v - min) / range) * (h - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const stroke = up ? "#34d399" : "#f87171";
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-12" preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={stroke} strokeWidth="1.5"
        strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// Compact order book — crypto only (Alpaca free tier has no L2 for stocks).
function MiniOrderBook({ symbol }: { symbol: string }) {
  const [book, setBook] = React.useState<{ bids: any[]; asks: any[] } | null>(null);
  React.useEffect(() => {
    if (isStock(symbol)) return;
    let cancelled = false;
    const load = () => {
      fetch(`${API_BASE}/market/depth?symbol=${encodeURIComponent(symbol)}&limit=6`)
        .then(r => r.json())
        .then(d => { if (!cancelled) setBook({ bids: d.bids || [], asks: d.asks || [] }); })
        .catch(() => { });
    };
    load();
    const id = setInterval(load, 6000);
    return () => { cancelled = true; clearInterval(id); };
  }, [symbol]);

  if (isStock(symbol)) {
    return <div className="text-[10px] text-zinc-600 italic py-2">Order book N/A (free stock feed has no L2 depth)</div>;
  }
  if (!book) return <div className="text-[10px] text-zinc-600 py-2">loading book…</div>;
  const asks = (book.asks || []).slice(0, 5).reverse();
  const bids = (book.bids || []).slice(0, 5);
  return (
    <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
      <div>
        <div className="text-zinc-500 mb-1 uppercase tracking-wider">Bids</div>
        {bids.map((b: any, i: number) => (
          <div key={i} className="flex justify-between text-emerald-400/90">
            <span>{fmtNum(parseFloat(b[0]), 2)}</span><span className="text-zinc-500">{fmtCompact(parseFloat(b[1]))}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="text-zinc-500 mb-1 uppercase tracking-wider">Asks</div>
        {asks.map((a: any, i: number) => (
          <div key={i} className="flex justify-between text-rose-400/90">
            <span>{fmtNum(parseFloat(a[0]), 2)}</span><span className="text-zinc-500">{fmtCompact(parseFloat(a[1]))}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AssetCard({ a }: { a: AssetRow }) {
  const up = (a.price_change_pct ?? 0) >= 0;
  const sb = sessionBadge(a.session);
  return (
    <Card className="p-4 flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base font-semibold text-zinc-100">{displaySymbol(a.symbol)}</span>
          <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-zinc-700/60 text-zinc-400">
            {a.asset_class === "crypto" ? "Crypto" : "Stock"}
          </span>
        </div>
        <span className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded border ${sb.cls}`}>{sb.label}</span>
      </div>

      {/* Price + change */}
      <div className="flex items-end justify-between">
        <div className="text-xl font-mono font-semibold text-zinc-100">
          ${fmtNum(a.last_price, a.last_price && a.last_price < 10 ? 4 : 2)}
        </div>
        <div className={`text-sm font-mono ${up ? "text-emerald-400" : "text-rose-400"}`}>
          {up ? "+" : ""}{fmtNum(a.price_change_pct, 2)}%
        </div>
      </div>

      <Sparkline data={a.spark} up={up} />

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div className="bg-[#0A0A0A] rounded p-2 border border-zinc-800/50">
          <div className="text-zinc-500 uppercase tracking-wider text-[9px]">Volatility</div>
          <div className="font-mono text-zinc-200">{fmtNum(a.volatility_pct, 2)}%</div>
        </div>
        <div className="bg-[#0A0A0A] rounded p-2 border border-zinc-800/50">
          <div className="text-zinc-500 uppercase tracking-wider text-[9px]">Volume</div>
          <div className="font-mono text-zinc-200">{fmtCompact(a.volume)}</div>
        </div>
        <div className="bg-[#0A0A0A] rounded p-2 border border-zinc-800/50">
          <div className="text-zinc-500 uppercase tracking-wider text-[9px]">24h</div>
          <div className={`font-mono ${up ? "text-emerald-400" : "text-rose-400"}`}>{up ? "+" : ""}{fmtNum(a.price_change_pct, 1)}%</div>
        </div>
      </div>

      {/* Extended-hours quote (stocks in pre/after) */}
      {a.extended_hours && (
        <div className="text-[10px] flex items-center justify-between px-2 py-1 rounded bg-amber-500/5 border border-amber-500/20">
          <span className="text-amber-400 uppercase tracking-wider">{a.extended_hours.session} px</span>
          <span className="font-mono text-zinc-200">${fmtNum(a.extended_hours.price, 2)}</span>
          <span className="text-zinc-600">via {a.extended_hours.source}</span>
        </div>
      )}

      {/* Order book */}
      <div className="border-t border-zinc-800/50 pt-2">
        <MiniOrderBook symbol={a.symbol} />
      </div>

      {/* Position */}
      <div className="border-t border-zinc-800/50 pt-2">
        {a.position ? (
          <div className="flex items-center justify-between text-[11px]">
            <span className={`uppercase font-bold ${a.position.direction === "short" ? "text-rose-400" : "text-emerald-400"}`}>
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

export default function AllAssetsPage() {
  const [assets, setAssets] = React.useState<AssetRow[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [err, setErr] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetch(`${API_BASE}/assets/overview`)
        .then(r => r.json())
        .then(d => {
          if (cancelled) return;
          setAssets(Array.isArray(d?.assets) ? d.assets : []);
          setErr(null);
        })
        .catch(e => { if (!cancelled) setErr(String(e)); })
        .finally(() => { if (!cancelled) setLoading(false); });
    };
    load();
    const id = setInterval(load, 10000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const crypto = assets.filter(a => a.asset_class === "crypto");
  const stocks = assets.filter(a => a.asset_class !== "crypto");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">All Assets</h1>
          <p className="text-sm text-zinc-500">Live cross-asset overview — graphs, order books, volatility, volume, sessions & positions.</p>
        </div>
        {loading && <span className="text-xs text-zinc-500">loading…</span>}
      </div>

      {err && <div className="text-sm text-rose-400">Failed to load: {err}</div>}

      {crypto.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-xs uppercase tracking-widest text-zinc-500">Crypto · 24/7</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {crypto.map(a => <AssetCard key={a.symbol} a={a} />)}
          </div>
        </section>
      )}

      {stocks.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-xs uppercase tracking-widest text-zinc-500">US Stocks</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {stocks.map(a => <AssetCard key={a.symbol} a={a} />)}
          </div>
        </section>
      )}

      {!loading && assets.length === 0 && !err && (
        <div className="text-zinc-500 text-sm">No asset data available.</div>
      )}
    </div>
  );
}
