"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";

export function SetupModal() {
  const pathname = usePathname();
  const [data, setData] = useState<{ needs_setup: boolean } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [skipped, setSkipped] = useState(false);
  const [step, setStep] = useState(0);

  const [provider, setProvider] = useState("gemini");
  const [llmKey, setLlmKey] = useState("");
  
  const [geminiKeyCached, setGeminiKeyCached] = useState("");
  const [anthropicKeyCached, setAnthropicKeyCached] = useState("");

  // DeFi
  const [arbitrumRpcUrl, setArbitrumRpcUrl] = useState("");
  const [agentPrivateKey, setAgentPrivateKey] = useState("");
  const [agentWalletAddress, setAgentWalletAddress] = useState("");
  
  // APIs
  const [alpacaApiKey, setAlpacaApiKey] = useState("");
  const [telegramApiId, setTelegramApiId] = useState("");
  const [xApiKey, setXApiKey] = useState("");

  useEffect(() => {
    // Check if skipped
    if (typeof window !== "undefined" && localStorage.getItem("setupSkipped") === "true") {
      setSkipped(true);
    }

    Promise.all([
      fetch("http://localhost:8000/api/setup/status").then(res => res.json()),
      fetch("http://localhost:8000/api/setup/config").then(res => res.json()).catch(() => ({}))
    ])
      .then(([statusData, configData]) => {
        setData(statusData);
        if (configData) {
          const isRealKey = (k: string) => !!(k && k.length > 5 && !k.toLowerCase().includes("your_") && !k.toLowerCase().includes("0x000"));
          const sanitize = (k: string) => isRealKey(k) ? k : "";

          if (configData.AI_PROVIDER) setProvider(configData.AI_PROVIDER);
          
          let gKey = sanitize(configData.GEMINI_API_KEY || "");
          let aKey = sanitize(configData.ANTHROPIC_API_KEY || "");
          setGeminiKeyCached(gKey);
          setAnthropicKeyCached(aKey);

          if (configData.AI_PROVIDER === "anthropic" && aKey) {
             setLlmKey(aKey);
          } else if (gKey) {
             setLlmKey(gKey);
          }
          if (configData.ARBITRUM_RPC_URL) setArbitrumRpcUrl(sanitize(configData.ARBITRUM_RPC_URL));
          if (configData.AGENT_PRIVATE_KEY) setAgentPrivateKey(sanitize(configData.AGENT_PRIVATE_KEY));
          if (configData.AGENT_WALLET_ADDRESS) setAgentWalletAddress(sanitize(configData.AGENT_WALLET_ADDRESS));
          if (configData.ALPACA_API_KEY) setAlpacaApiKey(sanitize(configData.ALPACA_API_KEY));
          if (configData.TELEGRAM_API_ID) setTelegramApiId(sanitize(configData.TELEGRAM_API_ID));
          if (configData.X_API_KEY) setXApiKey(sanitize(configData.X_API_KEY));
        }
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

  const handleProviderChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newProv = e.target.value;
    
    if (provider === "gemini") setGeminiKeyCached(llmKey);
    if (provider === "anthropic") setAnthropicKeyCached(llmKey);

    setProvider(newProv);
    setLlmKey(newProv === "gemini" ? geminiKeyCached : anthropicKeyCached);
  };

  const handleSave = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setSaving(true);
    setError(null);

    try {
      const payload = {
        AI_PROVIDER: provider,
        GEMINI_API_KEY: provider === "gemini" ? llmKey : geminiKeyCached,
        ANTHROPIC_API_KEY: provider === "anthropic" ? llmKey : anthropicKeyCached,
        ARBITRUM_RPC_URL: arbitrumRpcUrl,
        AGENT_PRIVATE_KEY: agentPrivateKey,
        AGENT_WALLET_ADDRESS: agentWalletAddress,
        ALPACA_API_KEY: alpacaApiKey,
        TELEGRAM_API_ID: telegramApiId,
        X_API_KEY: xApiKey,
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

  // Do not show any setup prompts or banners on the settings page itself
  if (pathname === '/settings') {
    return null;
  }

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
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 font-sans antialiased">
      <div className="bg-[#0C0C0E] border border-neutral-800/60 rounded-2xl max-w-sm w-full shadow-2xl overflow-hidden relative">
        <div className="absolute -top-32 -right-32 w-64 h-64 bg-neutral-500/5 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="p-6 relative z-10 min-h-[360px] flex flex-col">
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs p-3 rounded-lg mb-4 tracking-wide shrink-0">
              {error}
            </div>
          )}

          <div className="flex-grow flex flex-col justify-center">
            {step === 0 && (
              <div className="text-center animate-in fade-in slide-in-from-bottom-2 duration-300">
                <h2 className="text-xl tracking-tight font-semibold text-neutral-100 mb-3">Welcome to Trading Agent</h2>
                <p className="text-sm text-neutral-400 leading-relaxed mb-6">
                  Ready to get started? We just need to configure a few keys to initialize the engine.
                </p>
              </div>
            )}

            {step === 1 && (
              <div className="animate-in fade-in slide-in-from-right-4 duration-300 space-y-4">
                <div className="mb-6">
                  <h2 className="text-lg tracking-tight font-semibold text-neutral-100 mb-1">Intelligence</h2>
                  <p className="text-xs text-neutral-500">Language Model API Configuration</p>
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Provider</label>
                  <select
                    value={provider}
                    onChange={handleProviderChange}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none"
                  >
                    <option value="gemini">Google Gemini (Recommended)</option>
                    <option value="anthropic">Anthropic Claude</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">API Key</label>
                  <input
                    type="password"
                    value={llmKey}
                    onChange={(e) => setLlmKey(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder={`e.g. ${provider === 'gemini' ? 'AIza...' : 'sk-ant-...'}`}
                  />
                </div>
              </div>
            )}

            {step === 2 && (
              <div className="animate-in fade-in slide-in-from-right-4 duration-300 space-y-4">
                <div className="mb-6">
                  <h2 className="text-lg tracking-tight font-semibold text-neutral-100 mb-1">Web3 Execution</h2>
                  <p className="text-xs text-neutral-500">Configure your Arbitrum Wallet</p>
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Arbitrum RPC URL</label>
                  <input
                    type="text"
                    value={arbitrumRpcUrl}
                    onChange={(e) => setArbitrumRpcUrl(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder="RPC URL"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Agent Private Key</label>
                  <input
                    type="password"
                    value={agentPrivateKey}
                    onChange={(e) => setAgentPrivateKey(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder="0x..."
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Agent Wallet Address</label>
                  <input
                    type="text"
                    value={agentWalletAddress}
                    onChange={(e) => setAgentWalletAddress(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder="0x..."
                  />
                </div>
              </div>
            )}

            {step === 3 && (
              <div className="animate-in fade-in slide-in-from-right-4 duration-300 space-y-4">
                <div className="mb-6">
                  <h2 className="text-lg tracking-tight font-semibold text-neutral-100 mb-1">Social & Data</h2>
                  <p className="text-xs text-neutral-500">Optional integrations</p>
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Telegram API ID</label>
                  <input
                    type="text"
                    value={telegramApiId}
                    onChange={(e) => setTelegramApiId(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder="Telegram ID"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">X API Key</label>
                  <input
                    type="text"
                    value={xApiKey}
                    onChange={(e) => setXApiKey(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder="X API Key"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-neutral-400 mb-1.5 ml-1">Alpaca API Key</label>
                  <input
                    type="text"
                    value={alpacaApiKey}
                    onChange={(e) => setAlpacaApiKey(e.target.value)}
                    className="w-full bg-[#18181A] border border-[#333336] rounded-lg px-3 py-2 text-sm text-neutral-200 outline-none font-mono placeholder:font-sans"
                    placeholder="Alpaca Key"
                  />
                </div>
              </div>
            )}
          </div>

          <div className="pt-6 mt-auto shrink-0 flex flex-col gap-3">
            <div className="flex gap-3">
              {step > 0 && (
                <button
                  type="button"
                  onClick={() => setStep(step - 1)}
                  className="bg-neutral-800 hover:bg-neutral-700 text-neutral-300 font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
                >
                  Back
                </button>
              )}
              {step < 3 ? (
                <button
                  type="button"
                  onClick={() => setStep(step + 1)}
                  className="flex-1 bg-neutral-100 hover:bg-white text-black font-medium py-2.5 px-4 rounded-lg text-sm transition-colors"
                >
                  {step === 0 ? "Let's Go" : "Next"}
                </button>
              ) : (
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => handleSave()}
                  className="flex-1 bg-green-600 hover:bg-green-500 text-white font-medium py-2.5 px-4 rounded-lg text-sm transition-colors disabled:opacity-50"
                >
                  {saving ? "Saving..." : "Finish"}
                </button>
              )}
            </div>
            
            <button
              type="button"
              onClick={handleSkip}
              className="text-xs text-neutral-500 hover:text-neutral-300 transition-colors w-full text-center py-2"
            >
              Skip for now
            </button>
          </div>
          
          {/* Progress dots */}
          <div className="absolute top-4 right-4 flex gap-1">
            {[0, 1, 2, 3].map((s) => (
              <div
                key={s}
                className={`w-1.5 h-1.5 rounded-full transition-colors ${
                  step === s ? "bg-neutral-300" : "bg-neutral-800"
                }`}
              />
            ))}
          </div>

        </div>
      </div>
    </div>
  );
}
