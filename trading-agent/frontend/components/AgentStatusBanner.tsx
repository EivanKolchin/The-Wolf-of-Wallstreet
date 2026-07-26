"use client";

import { useEffect, useState } from "react";
import { Card } from "./ui/card";
import { Power, Play } from "lucide-react";
import { cn } from "@/lib/utils";

interface AgentStatus {
  is_halted: boolean;
  buffer_current: number;
  buffer_required: number;
  cycle_interval: number;
  started_at: number;
  has_market_data: boolean;
  status_text: string;
  /** "strategy" = the rule-based paper book (no warm-up buffer); absent for the NN engine. */
  engine?: string;
}

type EngineState = "connecting" | "offline" | "halted" | "warming" | "active";

function fmtUptime(sec: number) {
  if (!isFinite(sec) || sec < 0) return "0m";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.floor(sec)}s`;
}

export function AgentStatusBanner({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(false);
  const [now, setNow] = useState(Date.now() / 1000);

  const fetchStatus = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/agent/status");
      if (res.ok) { setStatus(await res.json()); setReachable(true); }
      else setReachable(false);
    } catch {
      setReachable(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => { clearInterval(interval); clearInterval(tick); };
  }, []);

  const toggleStop = async (halt: boolean) => {
    setLoading(true);
    try {
      await fetch("http://localhost:8000/api/agent/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ halt }),
      });
      await fetchStatus();
    } catch (err) {
      console.error("Failed to toggle agent state", err);
    } finally {
      setLoading(false);
    }
  };

  // ── Derive a single, honest engine state ──
  const warmupWindowSeconds = 300;
  const elapsedSinceStart = status ? Math.max(0, now - (status.started_at || now)) : 0;
  // The rule-based strategy book has no market-data buffer to fill (buffer 1/1), so it is Active
  // the moment it publishes — only the NN engine needs the 300s warm-up window. Gating on this
  // lets the backend report an HONEST started_at (uptime from 0) instead of a fake far-past value.
  const needsWarmup = status?.engine !== "strategy";
  const isWarmingUp = !!status &&
    !status.is_halted &&
    (status.buffer_current < status.buffer_required ||
      (needsWarmup && elapsedSinceStart < warmupWindowSeconds));

  let engine: EngineState = "connecting";
  if (reachable === false) engine = "offline";
  else if (status?.is_halted) engine = "halted";
  else if (status && isWarmingUp) engine = "warming";
  else if (status) engine = "active";

  const meta: Record<EngineState, { label: string; tone: string; sub: string }> = {
    connecting: { label: "Connecting", tone: "text-zinc-400", sub: "Reaching the trading engine…" },
    offline:    { label: "Offline",    tone: "text-zinc-500", sub: "Backend unreachable — engine not running." },
    halted:     { label: "Halted",     tone: "text-rose-400", sub: "Stopped by the kill switch. No orders will route." },
    warming:    { label: "Initializing", tone: "text-zinc-200", sub: status?.status_text || "Buffering market data…" },
    active:     { label: "Active",     tone: "text-emerald-400", sub: status?.status_text || "Scanning markets and evaluating signals." },
  };
  const m = meta[engine];

  // warm-up progress + ETA
  let remainingSeconds = 0;
  if (status && isWarmingUp) {
    let calc = Math.max(0, warmupWindowSeconds - elapsedSinceStart);
    if (status.buffer_current > 0 && status.buffer_current < status.buffer_required) {
      calc = Math.max(calc, (status.buffer_required - status.buffer_current) * status.cycle_interval);
    }
    remainingSeconds = calc;
  }
  const mins = Math.floor(remainingSeconds / 60);
  const secs = Math.floor(remainingSeconds % 60);
  let progressPct = 100;
  if (status && status.buffer_current < status.buffer_required) {
    progressPct = Math.max(0, (status.buffer_current / (status.buffer_required || 1)) * 100);
  } else if (isWarmingUp) {
    progressPct = Math.max(0, Math.min(100, (elapsedSinceStart / warmupWindowSeconds) * 100));
  }

  const canKill = engine === "active" || engine === "warming";

  if (compact) {
    return (
      <Card className="min-w-[320px] p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.15em] text-zinc-500">Engine State</p>
            <div className="flex items-baseline gap-2">
              <span className={cn("text-lg font-semibold tracking-tight", m.tone)}>{m.label}</span>
              {engine === "active" && (
                <span className="text-[10px] font-medium text-zinc-500">up {fmtUptime(elapsedSinceStart)}</span>
              )}
            </div>
            <p className="mt-1 max-w-[180px] truncate text-[11px] text-zinc-500" title={m.sub}>{m.sub}</p>
          </div>
          <button
            type="button"
            onClick={() => toggleStop(engine !== "halted")}
            disabled={loading || engine === "offline" || engine === "connecting"}
            className={cn(
              "inline-flex shrink-0 items-center gap-2 rounded-lg px-3 py-2 text-[11px] font-semibold tracking-wide transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              engine === "halted"
                ? "bg-zinc-100 text-zinc-900 hover:bg-white"
                : "border border-rose-500/30 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20"
            )}
          >
            {engine === "halted" ? <><Play size={13} /> Resume</> : <><Power size={13} /> Kill Switch</>}
          </button>
        </div>
      </Card>
    );
  }

  return (
    <Card className="flex h-full flex-col justify-between p-5">
      <div className="flex items-start justify-between gap-4">
        {/* Engine state — functional, no decorative dot */}
        <div className="min-w-0">
          <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.15em] text-zinc-500">Engine</p>
          <div className="flex items-baseline gap-2.5">
            <span className={cn("text-2xl font-semibold tracking-tight", m.tone)}>{m.label}</span>
            {engine === "active" && (
              <span className="text-[11px] font-medium text-zinc-500">live · up {fmtUptime(elapsedSinceStart)}</span>
            )}
          </div>
          <p className="mt-1.5 truncate text-[12px] text-zinc-500" title={m.sub}>{m.sub}</p>
        </div>

        {/* Kill / resume control */}
        <button
          type="button"
          onClick={() => toggleStop(engine !== "halted")}
          disabled={loading || engine === "offline" || engine === "connecting"}
          className={cn(
            "inline-flex shrink-0 items-center gap-2 rounded-lg px-4 py-2.5 text-[12px] font-semibold tracking-wide transition-colors disabled:cursor-not-allowed disabled:opacity-40",
            engine === "halted"
              ? "bg-zinc-100 text-zinc-900 hover:bg-white"
              : "border border-rose-500/30 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20"
          )}
        >
          {engine === "halted" ? <><Play size={14} /> Resume Engine</> : <><Power size={14} /> Force Kill</>}
        </button>
      </div>

      {/* Warm-up detail — only while initializing */}
      {engine === "warming" && (
        <div className="mt-5 grid grid-cols-2 gap-4 border-t border-[#171717] pt-4">
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-zinc-500">Data buffer</p>
              <p className="font-mono text-[13px] tabular-nums text-zinc-300">
                {status?.buffer_current} <span className="text-zinc-600">/ {status?.buffer_required}</span>
              </p>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
              <div className="h-full rounded-full bg-zinc-100 transition-all duration-500 ease-in-out" style={{ width: `${progressPct}%` }} />
            </div>
          </div>
          <div>
            <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.15em] text-zinc-500">Time to active</p>
            <p className="font-mono text-xl font-medium tabular-nums tracking-tight text-zinc-200">
              {mins}m {secs.toString().padStart(2, "0")}s
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}
