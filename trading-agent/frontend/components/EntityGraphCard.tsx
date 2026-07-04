"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { RefreshCcw } from "lucide-react";

interface GNode {
  id: string;
  kind: "crypto" | "equity";
}
interface GEdge {
  src: string;
  dst: string;
  kind: string;
  prior: number;
  weight: number; // correlation-validated LIVE weight; 0 = dead narrative
}
interface GraphPayload {
  active: boolean;
  nodes: GNode[];
  edges: GEdge[];
  updated_at?: number;
}

interface P {
  x: number;
  y: number;
}

const W = 560;
const H = 340;

/** Deterministic force layout (spring edges + node repulsion), iterated once per payload —
 *  small graphs (<40 nodes) converge in a few hundred steps, no animation loop needed. */
function layout(nodes: GNode[], edges: GEdge[]): Record<string, P> {
  const pos: Record<string, P> = {};
  const n = nodes.length;
  nodes.forEach((nd, i) => {
    const a = (2 * Math.PI * i) / Math.max(n, 1);
    pos[nd.id] = { x: W / 2 + 120 * Math.cos(a), y: H / 2 + 110 * Math.sin(a) };
  });
  const idx = Object.fromEntries(nodes.map((nd) => [nd.id, nd]));
  for (let step = 0; step < 300; step++) {
    const force: Record<string, P> = {};
    nodes.forEach((nd) => (force[nd.id] = { x: 0, y: 0 }));
    // repulsion
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const a = pos[nodes[i].id], b = pos[nodes[j].id];
        let dx = a.x - b.x, dy = a.y - b.y;
        const d2 = Math.max(dx * dx + dy * dy, 100);
        const f = 5200 / d2;
        const d = Math.sqrt(d2);
        dx /= d; dy /= d;
        force[nodes[i].id].x += dx * f; force[nodes[i].id].y += dy * f;
        force[nodes[j].id].x -= dx * f; force[nodes[j].id].y -= dy * f;
      }
    // springs (live edges pull harder than dead ones)
    edges.forEach((e) => {
      if (!idx[e.src] || !idx[e.dst]) return;
      const a = pos[e.src], b = pos[e.dst];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const rest = 120;
      const k = 0.02 + 0.06 * Math.min(1, Math.abs(e.weight));
      const f = k * (d - rest);
      force[e.src].x += (dx / d) * f; force[e.src].y += (dy / d) * f;
      force[e.dst].x -= (dx / d) * f; force[e.dst].y -= (dy / d) * f;
    });
    // centering + integrate + clamp
    nodes.forEach((nd) => {
      const p = pos[nd.id];
      force[nd.id].x += (W / 2 - p.x) * 0.005;
      force[nd.id].y += (H / 2 - p.y) * 0.005;
      p.x = Math.min(W - 40, Math.max(40, p.x + force[nd.id].x * 0.25));
      p.y = Math.min(H - 28, Math.max(28, p.y + force[nd.id].y * 0.25));
    });
  }
  return pos;
}

export function EntityGraphCard() {
  const [data, setData] = useState<GraphPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    async function fetchGraph() {
      try {
        const res = await fetch("http://localhost:8000/api/strategy/graph");
        if (!res.ok) throw new Error("failed");
        setData(await res.json());
      } catch {
        /* keep last */
      } finally {
        setLoading(false);
      }
    }
    fetchGraph();
    const id = setInterval(fetchGraph, 15000);
    return () => clearInterval(id);
  }, []);

  const pos = useMemo(
    () => (data?.nodes?.length ? layout(data.nodes, data.edges) : {}),
    [data]
  );

  if (loading && !data) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle>Cross-Asset Propagation Network</CardTitle>
          <RefreshCcw className="w-4 h-4 animate-spin text-zinc-500" />
        </CardHeader>
        <CardContent className="h-40 flex items-center justify-center">
          <p className="text-zinc-500">Loading graph…</p>
        </CardContent>
      </Card>
    );
  }

  const active = data?.active && (data?.nodes?.length ?? 0) > 0;
  const liveEdges = data?.edges?.filter((e) => e.weight > 0).length ?? 0;

  return (
    <Card>
      <CardHeader className="pb-3 border-b border-zinc-800/50 flex flex-row items-center justify-between">
        <CardTitle>Cross-Asset Propagation Network</CardTitle>
        <span className="text-[10px] uppercase tracking-widest text-zinc-600 font-mono">
          {liveEdges} live / {data?.edges?.length ?? 0} edges
        </span>
      </CardHeader>
      <CardContent className="pt-4">
        {!active ? (
          <div className="h-40 flex flex-col items-center justify-center gap-1 text-center">
            <p className="text-zinc-400 text-sm">No graph published yet.</p>
            <p className="text-zinc-600 text-xs font-mono">
              Appears after the strategy agent&apos;s first rebalance (news overlay on).
            </p>
          </div>
        ) : (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 380 }}>
              <defs>
                <marker id="arrow-live" viewBox="0 0 10 10" refX="9" refY="5"
                        markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#34d399" />
                </marker>
                <marker id="arrow-dead" viewBox="0 0 10 10" refX="9" refY="5"
                        markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#3f3f46" />
                </marker>
              </defs>
              {data!.edges.map((e, i) => {
                const a = pos[e.src], b = pos[e.dst];
                if (!a || !b) return null;
                const live = e.weight > 0;
                const dim = hover !== null && hover !== e.src && hover !== e.dst;
                // shorten the line so the arrowhead lands on the node ring
                const dx = b.x - a.x, dy = b.y - a.y;
                const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
                const tx = b.x - (dx / d) * 20, ty = b.y - (dy / d) * 20;
                return (
                  <g key={i} opacity={dim ? 0.15 : 1}>
                    <line
                      x1={a.x} y1={a.y} x2={tx} y2={ty}
                      stroke={live ? "#34d399" : "#3f3f46"}
                      strokeOpacity={live ? 0.35 + 0.6 * Math.min(1, e.weight) : 0.5}
                      strokeWidth={live ? 1 + 4 * Math.min(1, e.weight) : 1}
                      strokeDasharray={live ? undefined : "4 4"}
                      markerEnd={`url(#arrow-${live ? "live" : "dead"})`}
                    />
                    {live && !dim && (
                      <text x={(a.x + tx) / 2} y={(a.y + ty) / 2 - 4}
                            fontSize="8" fill="#34d399" opacity="0.8" textAnchor="middle"
                            fontFamily="monospace">
                        {e.weight.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}
              {data!.nodes.map((nd) => {
                const p = pos[nd.id];
                if (!p) return null;
                const dim = hover !== null && hover !== nd.id &&
                  !data!.edges.some(
                    (e) => (e.src === hover && e.dst === nd.id) ||
                           (e.dst === hover && e.src === nd.id));
                const label = nd.id.replace("USDT", "");
                return (
                  <g key={nd.id} opacity={dim ? 0.25 : 1} style={{ cursor: "pointer" }}
                     onMouseEnter={() => setHover(nd.id)}
                     onMouseLeave={() => setHover(null)}>
                    <circle cx={p.x} cy={p.y} r={14}
                            fill="#0e0e10"
                            stroke={nd.kind === "crypto" ? "#f4f4f5" : "#a1a1aa"}
                            strokeWidth={1.2} />
                    <text x={p.x} y={p.y + 3} fontSize="8.5" fill="#d4d4d8"
                          textAnchor="middle" fontFamily="monospace" fontWeight={600}>
                      {label.length > 5 ? label.slice(0, 5) : label}
                    </text>
                  </g>
                );
              })}
            </svg>
            <div className="mt-3 pt-3 border-t border-zinc-800/50 flex items-center gap-5 text-[10px] font-mono text-zinc-500 uppercase tracking-wider">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-4 h-0.5 bg-emerald-400" /> live (corr-validated)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-4 border-t border-dashed border-zinc-600" /> dead narrative
              </span>
              <span className="ml-auto normal-case">arrow = direction of propagation</span>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
