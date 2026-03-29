"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Save, Eye, EyeOff } from "lucide-react";
import Link from "next/link";

export default function SettingsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [provider, setProvider] = useState("gemini");
  const [llmKey, setLlmKey] = useState("");
  const [llmKeyLocked, setLlmKeyLocked] = useState(false);
  const [showLlmKey, setShowLlmKey] = useState(false);
  
  // Track specific AI keys to restore them if toggled
  const [geminiKeyCached, setGeminiKeyCached] = useState("");
  const [anthropicKeyCached, setAnthropicKeyCached] = useState("");
  
  // DeFi / App Config
  const [arbitrumRpcUrl, setArbitrumRpcUrl] = useState("");
  const [agentPrivateKey, setAgentPrivateKey] = useState("");
  const [agPkLocked, setAgPkLocked] = useState(false);
  const [showAgPk, setShowAgPk] = useState(false);
  const [agentWalletAddress, setAgentWalletAddress] = useState("");
  const [paperMode, setPaperMode] = useState(true);

  // Other Brokers / Social Services
  const [alpacaApiKey, setAlpacaApiKey] = useState("");
  const [alpacaSecretKey, setAlpacaSecretKey] = useState("");
  const [showAlpacaSecret, setShowAlpacaSecret] = useState(false);
  
  const [xApiKey, setXApiKey] = useState("");
  const [xApiSecret, setXApiSecret] = useState("");
  const [showXSecret, setShowXSecret] = useState(false);
  
  const [xAccessToken, setXAccessToken] = useState("");
  const [showXToken, setShowXToken] = useState(false);
  
  const [xAccessTokenSecret, setXAccessTokenSecret] = useState("");
  const [showXTokenSecret, setShowXTokenSecret] = useState(false);

  const [telegramApiId, setTelegramApiId] = useState("");
  const [telegramApiHash, setTelegramApiHash] = useState("");
  
  const [kiteApiKey, setKiteApiKey] = useState("");
  const [kiteApiSecret, setKiteApiSecret] = useState("");

  useEffect(() => {
    const fetchConfig = async (retries = 3) => {
      try {
        const res = await fetch(`http://localhost:8000/api/setup/config?t=${Date.now()}`);
        if (!res.ok) throw new Error("Backend not reachable");
        const cfg = await res.json();
        
        setProvider(cfg.AI_PROVIDER || "gemini");
        
        let gKey = cfg.GEMINI_API_KEY || "";
        let aKey = cfg.ANTHROPIC_API_KEY || "";
        
        setGeminiKeyCached(gKey);
        setAnthropicKeyCached(aKey);

        let initialLlmKey = "";
        if (cfg.AI_PROVIDER === "anthropic") {
           initialLlmKey = aKey;
        } else {
           initialLlmKey = gKey;
        }
        
        const isRealKey = (k: string) => !!(k && k.length > 5 && !k.toLowerCase().includes("your_") && !k.toLowerCase().includes("0x000"));
        const sanitize = (k: string) => isRealKey(k) ? k : "";

        setLlmKey(initialLlmKey);
        if (isRealKey(initialLlmKey)) setLlmKeyLocked(true);
        
        setPaperMode(cfg.PAPER_MODE === "true" || cfg.PAPER_MODE === true);

        setAgentPrivateKey(sanitize(cfg.AGENT_PRIVATE_KEY || ""));
        if (isRealKey(cfg.AGENT_PRIVATE_KEY)) setAgPkLocked(true);
        
        setAgentWalletAddress(sanitize(cfg.AGENT_WALLET_ADDRESS || ""));
        
        setAlpacaApiKey(sanitize(cfg.ALPACA_API_KEY || ""));
        setAlpacaSecretKey(sanitize(cfg.ALPACA_SECRET_KEY || ""));
        
        setXApiKey(sanitize(cfg.X_API_KEY || ""));
        setXApiSecret(sanitize(cfg.X_API_SECRET || ""));
        setXAccessToken(sanitize(cfg.X_ACCESS_TOKEN || ""));
        setXAccessTokenSecret(sanitize(cfg.X_ACCESS_TOKEN_SECRET || ""));
        
        setTelegramApiId(sanitize(cfg.TELEGRAM_API_ID || ""));
        setTelegramApiHash(sanitize(cfg.TELEGRAM_API_HASH || ""));
        
        setLoading(false);
      } catch (err) {
        if (retries > 0) {
          setTimeout(() => fetchConfig(retries - 1), 1000);
        } else {
          console.error("Failed to fetch setup config", err);
          setLoading(false);
        }
      }
    };
    
    fetchConfig();
  }, []);

  const handleProviderChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newProv = e.target.value;
    
    // Save current typed key into cache before switching
    if (provider === "gemini" && !llmKeyLocked) setGeminiKeyCached(llmKey);
    if (provider === "anthropic" && !llmKeyLocked) setAnthropicKeyCached(llmKey);

    setProvider(newProv);
    
    // Load the matching cached key
    const targetKey = newProv === "gemini" ? geminiKeyCached : anthropicKeyCached;
    setLlmKey(targetKey);
    const isRealKey = (k: string) => !!(k && k.length > 5 && !k.toLowerCase().includes("your_") && !k.toLowerCase().includes("0x000"));
    setLlmKeyLocked(isRealKey(targetKey));
    setShowLlmKey(false);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);

    try {
      const payload = {
        AI_PROVIDER: provider,
        GEMINI_API_KEY: provider === "gemini" ? llmKey : geminiKeyCached,
        ANTHROPIC_API_KEY: provider === "anthropic" ? llmKey : anthropicKeyCached,
        PAPER_MODE: paperMode ? "true" : "false",
        ARBITRUM_RPC_URL: arbitrumRpcUrl,
        AGENT_PRIVATE_KEY: agentPrivateKey,
        AGENT_WALLET_ADDRESS: agentWalletAddress,
        ALPACA_API_KEY: alpacaApiKey,
        ALPACA_SECRET_KEY: alpacaSecretKey,
        X_API_KEY: xApiKey,
        X_API_SECRET: xApiSecret,
        X_ACCESS_TOKEN: xAccessToken,
        X_ACCESS_TOKEN_SECRET: xAccessTokenSecret,
        TELEGRAM_API_ID: telegramApiId,
        TELEGRAM_API_HASH: telegramApiHash,
        KITE_API_KEY: kiteApiKey,
        KITE_API_SECRET: kiteApiSecret,
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
                  onChange={handleProviderChange}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-neutral-500/50 transition-all appearance-none"
                >
                  <option value="gemini">Google Gemini (GenAI Core)</option>
                  <option value="anthropic">Anthropic Claude (Alternative)</option>
                </select>
              </div>

              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Secret Authentication Key</label>
                <div className="flex gap-3">
                  <div className="relative w-full">
                    <input
                      type={showLlmKey ? "text" : "password"}
                      value={llmKey}
                      onChange={(e) => !llmKeyLocked && setLlmKey(e.target.value)}
                      disabled={llmKeyLocked}
                      className={`w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 ${llmKeyLocked ? 'pr-10 text-neutral-500 opacity-70' : 'text-neutral-200'} text-[14px] focus:outline-none focus:border-neutral-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700`}
                      placeholder={provider === 'gemini' ? 'AIza...' : 'sk-ant-...'}
                    />
                    {llmKeyLocked && (
                      <button
                        type="button"
                        onClick={() => setShowLlmKey(!showLlmKey)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
                      >
                        {showLlmKey ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    )}
                  </div>
                  {llmKeyLocked ? (
                    <button
                      type="button"
                      onClick={() => { setLlmKeyLocked(false); setLlmKey(""); setShowLlmKey(false); }}
                      className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-sm transition-colors whitespace-nowrap"
                    >
                      Update Key
                    </button>
                  ) : (
                    llmKey === "" && <div className="px-4 py-2 opacity-0 pointer-events-none whitespace-nowrap">Update Key</div> // spacer
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#222224] border border-[#333336] rounded-[16px] p-8 shadow-2xl">
            <h3 className="text-[11px] uppercase tracking-widest text-neutral-500 font-semibold mb-6 flex items-center">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500/80 mr-2"></div>
              Trading Mode
            </h3>

            <div className="space-y-4">
              <div className="flex items-center justify-between bg-[#161616] border border-[#333336] rounded-lg p-4">
                <div>
                  <h4 className="text-[14px] text-neutral-200 font-medium">Paper Trading</h4>
                  <p className="text-[12px] text-neutral-500 mt-1">Simulate trades without using real funds.</p>
                </div>
                <button
                  type="button"
                  onClick={() => setPaperMode(!paperMode)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${paperMode ? 'bg-[#FACC15]' : 'bg-neutral-600'} shrink-0`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${paperMode ? 'translate-x-6' : 'translate-x-1'}`} />
                </button>
              </div>
              {!paperMode && (
                <div className="p-3 rounded bg-red-500/10 border border-red-500/20 text-red-500 text-[12px] font-medium">
                  WARNING: Live trading is active. Real swaps will be executed on Arbitrum. Ensure your agent wallet has sufficient USDC balance and gas.
                </div>
              )}
            </div>
          </div>

          <div className="bg-[#222224] border border-[#333336] rounded-[16px] p-8 shadow-2xl">
            <h3 className="text-[11px] uppercase tracking-widest text-neutral-500 font-semibold mb-6 flex items-center">
              <div className="w-1.5 h-1.5 rounded-full bg-orange-500/80 mr-2"></div>
              Execution Environment (DeFi & Web3)
            </h3>
            
            <div className="space-y-5">
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Arbitrum RPC URL</label>
                <input
                  type="text"
                  value={arbitrumRpcUrl}
                  onChange={(e) => setArbitrumRpcUrl(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-orange-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="https://arb-mainnet.g.alchemy.com/v2/..."
                />
              </div>

              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Agent Private Key (Optional - Needed for Trades)</label>
                <div className="flex gap-3">
                  <div className="relative w-full">
                    <input
                      type={showAgPk ? "text" : "password"}
                      value={agentPrivateKey}
                      onChange={(e) => !agPkLocked && setAgentPrivateKey(e.target.value)}
                      disabled={agPkLocked}
                      className={`w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 ${agPkLocked ? 'pr-10 text-neutral-500 opacity-70' : 'text-neutral-200'} text-[14px] focus:outline-none focus:border-orange-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700`}
                      placeholder="0x..."
                    />
                    {agPkLocked && (
                      <button
                        type="button"
                        onClick={() => setShowAgPk(!showAgPk)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
                      >
                        {showAgPk ? <EyeOff size={16} /> : <Eye size={16} />}
                      </button>
                    )}
                  </div>
                  {agPkLocked ? (
                    <button
                      type="button"
                      onClick={() => { setAgPkLocked(false); setAgentPrivateKey(""); setShowAgPk(false); }}
                      className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 rounded-lg text-sm transition-colors whitespace-nowrap"
                    >
                      Update Key
                    </button>
                  ) : (
                    agentPrivateKey === "" && <div className="px-4 py-2 opacity-0 pointer-events-none whitespace-nowrap">Update Key</div> // spacer
                  )}
                </div>
              </div>

              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Agent Wallet Address</label>
                <input
                  type="text"
                  value={agentWalletAddress}
                  onChange={(e) => setAgentWalletAddress(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-orange-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="0x..."
                />
              </div>
            </div>
          </div>

          <div className="bg-[#222224] border border-[#333336] rounded-[16px] p-8 shadow-2xl">
            <h3 className="text-[11px] uppercase tracking-widest text-neutral-500 font-semibold mb-6 flex items-center">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500/80 mr-2"></div>
              Other Integrations
            </h3>

            <div className="space-y-5 mb-8">
              <h4 className="text-sm text-neutral-300 font-medium">Alpaca Markets</h4>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">API Key</label>
                <input
                  type="text"
                  value={alpacaApiKey}
                  onChange={(e) => setAlpacaApiKey(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                  placeholder="PK..."
                />
              </div>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">Secret Key</label>
                <div className="relative w-full">
                  <input
                    type={showAlpacaSecret ? "text" : "password"}
                    value={alpacaSecretKey}
                    onChange={(e) => setAlpacaSecretKey(e.target.value)}
                    className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 pr-10 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all font-mono placeholder:font-sans placeholder-neutral-700"
                    placeholder="Secret..."
                  />
                  {alpacaSecretKey && (
                    <button
                      type="button"
                      onClick={() => setShowAlpacaSecret(!showAlpacaSecret)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
                    >
                      {showAlpacaSecret ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-[#333] to-transparent my-6"></div>

            <div className="space-y-5 mb-8">
              <h4 className="text-sm text-neutral-300 font-medium">X (Twitter) Integration</h4>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">X API Key</label>
                <input
                  type="text"
                  value={xApiKey}
                  onChange={(e) => setXApiKey(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                  placeholder="Key"
                />
              </div>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">X API Secret</label>
                <div className="relative w-full">
                  <input
                    type={showXSecret ? "text" : "password"}
                    value={xApiSecret}
                    onChange={(e) => setXApiSecret(e.target.value)}
                    className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 pr-10 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                    placeholder="Secret"
                  />
                  {xApiSecret && (
                    <button
                      type="button"
                      onClick={() => setShowXSecret(!showXSecret)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
                    >
                      {showXSecret ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  )}
                </div>
              </div>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">X Access Token</label>
                <div className="relative w-full">
                  <input
                    type={showXToken ? "text" : "password"}
                    value={xAccessToken}
                    onChange={(e) => setXAccessToken(e.target.value)}
                    className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 pr-10 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                    placeholder="Token"
                  />
                  {xAccessToken && (
                    <button
                      type="button"
                      onClick={() => setShowXToken(!showXToken)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
                    >
                      {showXToken ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  )}
                </div>
              </div>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">X Access Token Secret</label>
                <div className="relative w-full">
                  <input
                    type={showXTokenSecret ? "text" : "password"}
                    value={xAccessTokenSecret}
                    onChange={(e) => setXAccessTokenSecret(e.target.value)}
                    className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 pr-10 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                    placeholder="Secret"
                  />
                  {xAccessTokenSecret && (
                    <button
                      type="button"
                      onClick={() => setShowXTokenSecret(!showXTokenSecret)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500 hover:text-neutral-300 transition-colors"
                    >
                      {showXTokenSecret ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-[#333] to-transparent my-6"></div>

            <div className="space-y-5">
              <h4 className="text-sm text-neutral-300 font-medium">Telegram Integrations</h4>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">API ID</label>
                <input
                  type="text"
                  value={telegramApiId}
                  onChange={(e) => setTelegramApiId(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                  placeholder="API ID"
                />
              </div>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">API Hash</label>
                <input
                  type="password"
                  value={telegramApiHash}
                  onChange={(e) => setTelegramApiHash(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                  placeholder="API Hash"
                />
              </div>
            </div>

            <div className="w-full h-[1px] bg-gradient-to-r from-transparent via-[#333] to-transparent my-6"></div>

            <div className="space-y-5">
              <h4 className="text-sm text-neutral-300 font-medium">Kite (Zerodha) Integrations</h4>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">API Key</label>
                <input
                  type="text"
                  value={kiteApiKey}
                  onChange={(e) => setKiteApiKey(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                  placeholder="Kite API Key"
                />
              </div>
              <div>
                <label className="block text-[13px] font-medium text-neutral-400 mb-2">API Secret</label>
                <input
                  type="password"
                  value={kiteApiSecret}
                  onChange={(e) => setKiteApiSecret(e.target.value)}
                  className="w-full bg-[#161616] border border-[#333336] rounded-lg px-4 py-3 text-[14px] text-neutral-200 focus:outline-none focus:border-blue-500/50 transition-all"
                  placeholder="Kite API Secret"
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
                  Save
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
