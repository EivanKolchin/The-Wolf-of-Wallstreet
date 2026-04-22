"use client";

import { useEffect, useState } from "react";
import { Card, CardContent } from "./ui/card";
import { Button } from "./ui/button";
import { cn } from "@/lib/utils";

interface AgentStatus {
  is_halted: boolean;
  buffer_current: number;
  buffer_required: number;
  cycle_interval: number;
  started_at: number;
  has_market_data: boolean;
  status_text: string;
}

export function AgentStatusBanner() {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [now, setNow] = useState(Date.now() / 1000);
  const [localStart, setLocalStart] = useState<number | null>(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/agent/status");
      if (res.ok) {
        const json = await res.json();
        setStatus(json);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000); // 1-second ticks
    return () => {
      clearInterval(interval);
      clearInterval(tick);
    };
  }, []);

  useEffect(() => {
    if (status && !localStart) {
      setLocalStart(status.started_at || Date.now() / 1000);
    }
  }, [status, localStart]);

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

  if (!status) return null;

  const isWarmingUp = ((now - (status.started_at * 1000) < 300000) || status.buffer_current < status.buffer_required) && !status.is_halted;
  
  // Real-time calculation mechanics
  let remainingSeconds = 0;
  if (isWarmingUp) {
     const elapsedSinceStart = (now - (status.started_at * 1000)) / 1000;
     
     // Start from 5 minutes (300 seconds) since that's what the UI usually requires to buffer
     let calculatedRemaining = Math.max(0, 300 - elapsedSinceStart);
     
     if (!status.has_market_data || status.buffer_current === 0) {
        // Still waiting for first kline. Keep ticking down but pause at 0 if no data
        if (calculatedRemaining < 0) calculatedRemaining = 0;
     } else if (status.buffer_current < status.buffer_required) {
        // Market data established!
        const bufferTimeRemaining = (status.buffer_required - status.buffer_current) * status.cycle_interval;
        calculatedRemaining = Math.max(calculatedRemaining, bufferTimeRemaining);
     }

     remainingSeconds = calculatedRemaining;
  }
  
  const mins = Math.floor(remainingSeconds / 60);
  const secs = Math.floor(remainingSeconds % 60);

  // Compute bar width safely
  let progressPct = 100;
  if (status.buffer_current < status.buffer_required) {
      progressPct = Math.max(0, (status.buffer_current / status.buffer_required) * 100);
  } else if (isWarmingUp) {
      const elapsed = (now - (status.started_at * 1000)) / 1000;
      progressPct = Math.max(0, Math.min(100, (elapsed / 300) * 100));
  }

  return (
    <Card className="col-span-2 border-zinc-800/50 bg-[#0A0A0A] overflow-hidden rounded-xl shadow-lg">
      <CardContent className="p-0 grid grid-cols-1 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-zinc-800/50">
        
        {/* Engine State */}
        <div className="p-5 flex flex-col justify-center bg-black/40">
          <p className="text-[11px] font-semibold text-zinc-500 tracking-wider mb-2">Engine State</p>
          <div className="flex items-center gap-2.5">
            {!isWarmingUp && (
              <div className={cn("w-2 h-2 rounded-full flex-shrink-0",
                status.is_halted ? "bg-rose-500" : "bg-emerald-500"
              )} />
            )}
            <span className={cn("text-sm font-semibold tracking-wide",
              status.is_halted ? "text-rose-500" :
              isWarmingUp ? "text-amber-500" :
              "text-emerald-500"
            )}>
              {status.is_halted ? "Terminated" : isWarmingUp ? "Initializing..." : "Active"}
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-1.5 truncate" title={status.status_text}>{status.status_text}</p>
        </div>

        {/* Data Buffer */}
        <div className="p-5 flex flex-col justify-center">
          <div className="flex justify-between items-center mb-1.5">
             <p className="text-[11px] font-semibold text-zinc-500 tracking-wider">Data Buffer</p>
             <p className="text-sm font-medium tabular-nums text-[#D1D4DC]">
                {status.buffer_current} <span className="text-zinc-600">/ {status.buffer_required}</span>
             </p>
          </div>
          <div className="w-full h-1.5 bg-zinc-900 rounded-full mt-2 overflow-hidden flex-shrink-0">
             <div 
               className={cn("h-full rounded-full transition-all duration-500 ease-in-out", 
                 status.is_halted ? "bg-rose-500/50" : 
                 isWarmingUp ? "bg-amber-500" : 
                 "bg-emerald-500"
               )}
               style={{ width: `${progressPct}%` }}
             />
          </div>
        </div>

        {/* Time To Active */}
        <div className="p-5 flex flex-col justify-center">
            <p className="text-[11px] font-semibold text-zinc-500 tracking-wider mb-1">Time To Active</p>
            <p className={cn("text-3xl font-medium tracking-tight tabular-nums mt-1", 
              status.is_halted ? "text-zinc-600" :
              isWarmingUp ? "text-amber-400" : 
              "text-emerald-500"
            )}>
              {status.is_halted ? "--:--" : isWarmingUp ? `${mins}m ${secs.toString().padStart(2, "0")}s` : "0m 00s"}
            </p>
        </div>

        {/* Action Panel */}
        <div className="p-4 flex items-center justify-center bg-black/20">
           {status.is_halted ? (
             <Button
                variant="outline"
                size="sm"
                onClick={() => toggleStop(false)}
                disabled={loading}
                className="w-full h-11 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 border-emerald-500/30 tracking-wider text-xs font-semibold transition-colors"
              >
                Resume Engine
              </Button>
          ) : (
             <Button
              variant="outline"
              size="sm"
              onClick={() => toggleStop(true)}
              disabled={loading}
              className="w-full h-11 bg-rose-500/10 hover:bg-rose-500/20 text-rose-500 border-rose-500/30 tracking-wider text-xs font-semibold transition-colors"
            >
              Force Kill Switch
            </Button>
          )}
        </div>

      </CardContent>
    </Card>
  );
}
