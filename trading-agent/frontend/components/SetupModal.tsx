"use client";

import { useEffect, useState } from "react";

export function SetupModal() {
  const [data, setData] = useState<{ needs_setup: boolean } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [skipped, setSkipped] = useState(false);

  const [provider, setProvider] = useState("gemini");
  const [llmKey, setLlmKey] = useState("");
  const [binanceKey, setBinanceKey] = useState("");
  const [binanceSecret, setBinanceSecret] = useState("");

  useEffect(() => {
    // Check if skipped
    if (typeof window !== "undefined" && localStorage.getItem("setupSkipped") === "true") {
      setSkipped(true);
    }

    fetch("http://localhost:8000/api/setup/status")
      .then((res) => {
        if (!res.ok) throw new Error("Backend not reachable");
        return res.json();
      })
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch setup status", err);
        setLoading(false);
        // Force setup screen if backend is unreachable 
        // fallback so user can at least try
        setData({ needs_setup: true });
      });
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);

    try {
      const payload = {
        AI_PROVIDER: provider,
        GEMINI_API_KEY: provider === "gemini" ? llmKey : "",
        ANTHROPIC_API_KEY: provider === "anthropic" ? llmKey : "",
        BINANCE_API_KEY: binanceKey,
        BINANCE_SECRET: binanceSecret,
      };

      const res = await fetch("http://localhost:8000/api/setup/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error("Failed to save configuration");
      }

      alert("Configuration saved. The backend will now reboot. Please wait a few seconds and refresh.");
      localStorage.removeItem("setupSkipped");
      window.location.reload();

    } catch (err: any) {
      setError(err.message || "Network error while saving setup");
    } finally {
      setSaving(false);
    }
  };

  const handleSkip = () => {
    localStorage.setItem("setupSkipped", "true");
    setSkipped(true);
    window.dispatchEvent(new Event("setupSkipped"));
  };

  if (loading || !data) {
    return null;
  }

  if (!data.needs_setup && !skipped) {
    return null; // All good!
  }

  if (skipped && data.needs_setup) {
    return (
      <div className="fixed bottom-0 left-0 top-auto border-t border-b-0 pointer-events-none w-full z-[100] bg-orange-950/40 border-b border-orange-500/20 backdrop-blur-md h-12 flex justify-center items-center text-xs font-medium tracking-widest text-orange-200/80 shadow-md">
        <span>AGENT OFFLINE: KEYS MISSING.</span>
        <a style={{ pointerEvents: 'auto' }} href="/settings" className="ml-3 underline decoration-orange-500/50 hover:text-orange-100 transition-colors">
          CONFIGURE SETTINGS
        </a>
      </div>
    );
  }

  return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xl p-4 font-sans antialiased">
      <div className="bg-[#222224] border border-[#333336] rounded-[14px] w-full max-w-lg shadow-[0_0_80px_rgba(0,0,0,0.8)] relative overflow-hidden">
        
        {/* Subtle glowing orb in background */}
        <div className="absolute -top-32 -right-32 w-64 h-64 bg-neutral-500/5 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="p-8 relative z-10">
          <div className="mb-8">
            <h2 className="text-2xl tracking-tight font-semibold text-neutral-100 mb-2">Initialize Core</h2>
            <p className="text-sm text-neutral-500 leading-relaxed">
              Trading engines require active marketplace connections and intelligence provider keys to function.
            </p>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs p-3 rounded-lg mb-6 tracking-wide">
              {error}
            </div>
          )}

          <form onSubmit={handleSave} className="space-y-6">
            
            <div className="space-y-4">
              <h3 className="text-xs uppercase tracking-widest text-neutral-600 font-semibold mb-2">Intelligence</h3>
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-4 py-3 text-sm text-neutral-200 focus:outline-none focus:border-neutral-500/50 focus:ring-1 focus:ring-neutral-500/20 transition-all appearance-none"
                >
                  <option value="gemini">Google Gemini (Recommended)</option>
                  <option value="anthropic">Anthropic Claude</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">API Key</label>
                <input
                  type="password"
                  required
                  value={llmKey}
                  onChange={(e) => setLlmKey(e.target.value)}
                  className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-4 py-3 text-sm text-neutral-200 focus:outline-none focus:border-neutral-500/50 focus:ring-1 focus:ring-neutral-500/20 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder={`e.g. ${provider === 'gemini' ? 'AIza...' : 'sk-ant-...'}`}
                />
              </div>
            </div>

            <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-[#222] to-transparent my-2"></div>

            <div className="space-y-4">
              <h3 className="text-xs uppercase tracking-widest text-neutral-600 font-semibold mb-2">Marketplace</h3>
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Binance API Key</label>
                <input
                  type="password"
                  required
                  value={binanceKey}
                  onChange={(e) => setBinanceKey(e.target.value)}
                  className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-4 py-3 text-sm text-neutral-200 focus:outline-none focus:border-neutral-500/50 focus:ring-1 focus:ring-neutral-500/20 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="64-character public key"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Binance Secret Key</label>
                <input
                  type="password"
                  required
                  value={binanceSecret}
                  onChange={(e) => setBinanceSecret(e.target.value)}
                  className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-4 py-3 text-sm text-neutral-200 focus:outline-none focus:border-neutral-500/50 focus:ring-1 focus:ring-neutral-500/20 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="64-character private secret"
                />
              </div>
            </div>

            <div className="pt-4 flex flex-col gap-4">
              <button
                type="submit"
                disabled={saving}
                className="w-full bg-neutral-100 hover:bg-white text-black font-medium py-3.5 px-4 rounded-lg text-sm transition-colors disabled:opacity-50 tracking-wide"
              >
                {saving ? "Authenticating..." : "Initialize Engine"}
              </button>
              
              <button
                type="button"
                onClick={handleSkip}
                className="text-xs text-neutral-600 hover:text-neutral-300 transition-colors w-full text-center"
              >
                Skip for now (Read-Only Mode)
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
