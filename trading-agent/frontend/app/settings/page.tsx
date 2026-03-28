"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Save } from "lucide-react";
import Link from "next/link";

export default function SettingsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [provider, setProvider] = useState("gemini");
  const [llmKey, setLlmKey] = useState("");
  const [binanceKey, setBinanceKey] = useState("");
  const [binanceSecret, setBinanceSecret] = useState("");

  useEffect(() => {
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
        setData({});
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

      alert("Settings saved successfully. The backend will reboot with new keys.");
      localStorage.removeItem("setupSkipped"); 
      window.location.href = "/dashboard";
    } catch (err: any) {
      setError(err.message || "Network error while saving settings");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-transparent text-neutral-500">
        <div className="inline-block animate-pulse">Initializing Interface...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-transparent text-white p-8 md:p-12 font-sans overflow-y-auto">
      <div className="max-w-3xl mx-auto mt-4 relative">
        {/* Decorative background flare */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-neutral-800/5 rounded-full blur-[120px] pointer-events-none"></div>

        <Link href="/dashboard" className="inline-flex items-center text-xs text-neutral-500 hover:text-neutral-200 transition-colors uppercase tracking-widest mb-8">
          <ArrowLeft size={14} className="mr-2" /> Return to Dashboard
        </Link>
        
        <h1 className="text-3xl font-medium tracking-tight mb-2">System Configuration</h1>
        <p className="text-sm text-neutral-500 mb-10 w-2/3 leading-relaxed">
          Manage your intelligence providers and execution credentials. Updates to these parameters will trigger an automatic core reboot.
        </p>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm p-4 rounded-lg mb-8 backdrop-blur-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-8 relative z-10 w-full max-w-xl">
          
          <div className="bg-[#222224] border border-[#333336] rounded-[16px] p-8 shadow-2xl">
            <h3 className="text-[11px] uppercase tracking-widest text-neutral-500 font-semibold mb-6 flex items-center">
              <div className="w-1.5 h-1.5 rounded-full bg-neutral-800/80 mr-2"></div>
              Intelligence Matrix
            </h3>
            
            <div className="space-y-5">
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">LLM Provider</label>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-neutral-500/50 transition-all appearance-none"
                >
                  <option value="gemini">Google Gemini (GenAI Core)</option>
                  <option value="anthropic">Anthropic Claude (Alternative)</option>
                </select>
              </div>

              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Secret Authentication Key</label>
                <input
                  type="password"
                  value={llmKey}
                  onChange={(e) => setLlmKey(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-neutral-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder={provider === 'gemini' ? 'AIza...' : 'sk-ant-...'}
                />
              </div>
            </div>
          </div>

          <div className="bg-[#222224] border border-[#333336] rounded-[16px] p-8 shadow-2xl">
            <h3 className="text-[11px] uppercase tracking-widest text-neutral-500 font-semibold mb-6 flex items-center">
              <div className="w-1.5 h-1.5 rounded-full bg-orange-500/80 mr-2"></div>
              Execution Environment
            </h3>
            
            <div className="space-y-5">
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Binance Access Key</label>
                <input
                  type="password"
                  value={binanceKey}
                  onChange={(e) => setBinanceKey(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-orange-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="Enter public key fragment"
                />
              </div>

              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Binance Signature Secret</label>
                <input
                  type="password"
                  value={binanceSecret}
                  onChange={(e) => setBinanceSecret(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-orange-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="Enter private signature token"
                />
              </div>
            </div>
          </div>

          <div className="flex items-center justify-end mt-4">
            <button
              type="submit"
              disabled={saving}
              className="bg-neutral-100 hover:bg-white text-black font-medium py-3 px-6 rounded-lg text-sm transition-all disabled:opacity-50 tracking-wide flex items-center"
            >
              {saving ? (
                "Synchronizing..."
              ) : (
                <>
                  <Save size={16} className="mr-2" />
                  Commit Changes
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
