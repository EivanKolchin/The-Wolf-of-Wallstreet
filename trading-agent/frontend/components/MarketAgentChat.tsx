"use client";

import React from "react";
import { Bot, Send, UserRound } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { API_BASE } from "@/lib/api";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  route?: string;
};

const STARTERS = [
  "What is the current price?",
  "Am I in paper trading?",
  "When does NVDA report earnings?",
  "What's the analyst consensus on BTC?",
];

function routeLabel(route: { intent?: string; tier?: string } | null | undefined): string | undefined {
  if (!route) return undefined;
  if (route.intent === "research") {
    return route.tier === "sonnet" ? "web research · gemini pro" : "web research · gemini flash";
  }
  return "local model";
}

export function MarketAgentChat({ symbol }: { symbol: string }) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Ask me about prices, PnL, positions, earnings, agent health, or why a subsystem looks stuck. I can search the web when local data is missing.",
    },
  ]);
  const [input, setInput] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [loadingHint, setLoadingHint] = React.useState("Reading internal state and asking the configured LLM...");
  const scrollRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const ask = async (text: string) => {
    const question = text.trim();
    if (!question || loading) return;
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: question }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setLoadingHint("Reading internal state and asking the configured LLM...");
    // Hard client-side timeout: a stalled backend (web-search / LLM hang) must never
    // leave the user staring at the loading dot with no reply.
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90_000);
    try {
      const res = await fetch(`${API_BASE}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: question,
          symbol,
          history: nextMessages.slice(-8),
        }),
        signal: controller.signal,
      });
      const data = await res.json();
      if (data?.web_search?.results?.length) {
        setLoadingHint(`Found ${data.web_search.results.length} web result(s) for: ${data.web_search.query}`);
      }
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: res.ok ? (data.answer || "No response.") : (data.detail || "Chat request failed."),
          route: res.ok ? routeLabel(data.route) : undefined,
        },
      ]);
    } catch (e: any) {
      const msg = e?.name === "AbortError"
        ? "The copilot took too long to respond (timed out after 90s). It may be fetching web results or the LLM provider is slow — try again."
        : `Chat request failed: ${e?.message || e}`;
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: msg },
      ]);
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  return (
    <Card className="flex min-h-[520px] flex-col overflow-hidden">
      <CardHeader className="border-b border-[#171717] pb-4">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-sm">Market Copilot</CardTitle>
          <span className="rounded-full border border-[#1f1f22] bg-[#0e0e10] px-2 py-0.5 font-mono text-[10px] text-zinc-500">
            {symbol.replace("USDT", "")}
          </span>
        </div>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col p-0">
        <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
          {messages.map((msg, i) => {
            const isUser = msg.role === "user";
            return (
              <div key={i} className={`flex gap-2 ${isUser ? "justify-end" : "justify-start"}`}>
                {!isUser && (
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-emerald-500/20 bg-emerald-500/10 text-emerald-300">
                    <Bot size={14} />
                  </div>
                )}
                <div className={`max-w-[85%] rounded-2xl px-3 py-2 text-[12px] leading-relaxed ${
                  isUser
                    ? "bg-zinc-100 text-zinc-900"
                    : "border border-[#1a1a1c] bg-[#0e0e10] text-zinc-300"
                }`}>
                  {msg.content}
                  {msg.route && (
                    <div className="mt-1.5 font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                      {msg.route}
                    </div>
                  )}
                </div>
                {isUser && (
                  <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-zinc-500/20 bg-zinc-500/10 text-zinc-300">
                    <UserRound size={14} />
                  </div>
                )}
              </div>
            );
          })}
          {loading && (
            <div className="flex items-center gap-2 text-[11px] text-zinc-500">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
              {loadingHint}
            </div>
          )}
        </div>

        <div className="border-t border-[#171717] p-3">
          <div className="mb-2 flex flex-wrap gap-1.5">
            {STARTERS.map(starter => (
              <button
                key={starter}
                type="button"
                onClick={() => ask(starter)}
                className="rounded-full border border-[#1f1f22] px-2.5 py-1 text-[10px] text-zinc-500 transition-colors hover:border-emerald-500/40 hover:text-emerald-300"
              >
                {starter}
              </button>
            ))}
          </div>
          <form
            className="flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask(input);
                }
              }}
              placeholder={`Ask about ${symbol.replace("USDT", "")}, PnL, earnings, or agent health...`}
              className="max-h-28 min-h-[44px] flex-1 resize-none rounded-xl border border-[#1f1f22] bg-[#0e0e10] px-3 py-2 text-[12px] text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-emerald-500/40"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-zinc-100 text-zinc-950 transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
              title="Ask copilot"
            >
              <Send size={15} />
            </button>
          </form>
        </div>
      </CardContent>
    </Card>
  );
}
