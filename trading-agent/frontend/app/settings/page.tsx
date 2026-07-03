"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, Save } from "lucide-react";
import Link from "next/link";
import { WalletPanel } from "@/components/WalletPanel";
import { Field, TextInput, SecretInput, SelectInput, Toggle, ControlRow, Section, Divider } from "@/components/ui/field";

// Secret env keys we prefill from the .env and never overwrite with a blank
// value on save — leaving one of these fields empty means "leave it as-is",
// so switching providers can't silently wipe another provider's key.
const SECRET_KEYS = [
  "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "AGENT_PRIVATE_KEY", "RAMP_HOST_API_KEY",
  "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "X_API_KEY", "X_API_SECRET",
  "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "TELEGRAM_API_ID", "TELEGRAM_API_HASH",
  "BINANCE_FUTURES_API_KEY", "BINANCE_FUTURES_SECRET",
];

export default function SettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);

  const [provider, setProvider] = useState("gemini");
  const [ollamaModel, setOllamaModel] = useState("llama3");

  const [geminiKey, setGeminiKey] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");

  // DeFi / App Config
  const [arbitrumRpcUrl, setArbitrumRpcUrl] = useState("");
  const [agentPrivateKey, setAgentPrivateKey] = useState("");
  const [agentWalletAddress, setAgentWalletAddress] = useState("");
  const [paperMode, setPaperMode] = useState(true);

  // ── Live trading: master switch + Binance futures venue + 2-sleeve StrategyAgent ──
  const [paperTrading, setPaperTrading] = useState(true);
  const [binanceKey, setBinanceKey] = useState("");
  const [binanceSecret, setBinanceSecret] = useState("");
  const [binanceTestnet, setBinanceTestnet] = useState(true);
  const [binanceLeverage, setBinanceLeverage] = useState("2");
  const [strategyEnabled, setStrategyEnabled] = useState(false);
  const [strategySymbols, setStrategySymbols] = useState("SPY QQQ TQQQ TLT GLD BTC-USD ETH-USD");
  const [tsSymbols, setTsSymbols] = useState("");
  const [wManaged, setWManaged] = useState("0.5");
  const [wTs, setWTs] = useState("0.5");
  const [newsOverlay, setNewsOverlay] = useState(true);
  const [newsLlmVerify, setNewsLlmVerify] = useState(false);
  const [maxOrderUsd, setMaxOrderUsd] = useState("5000.0");

  // Funding: WalletConnect projectId + Ramp on-ramp key (Workstreams D/E).
  const [walletConnectProjectId, setWalletConnectProjectId] = useState("");
  const [rampHostApiKey, setRampHostApiKey] = useState("");

  // Other Brokers / Social Services
  const [alpacaApiKey, setAlpacaApiKey] = useState("");
  const [alpacaSecretKey, setAlpacaSecretKey] = useState("");
  const [xApiKey, setXApiKey] = useState("");
  const [xApiSecret, setXApiSecret] = useState("");
  const [xAccessToken, setXAccessToken] = useState("");
  const [xAccessTokenSecret, setXAccessTokenSecret] = useState("");
  const [telegramApiId, setTelegramApiId] = useState("");
  const [telegramApiHash, setTelegramApiHash] = useState("");

  // Advanced risk options
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [riskMaxDrawdown, setRiskMaxDrawdown] = useState("15.0");
  const [riskMaxDailyLoss, setRiskMaxDailyLoss] = useState("5.0");
  const [riskMaxPosition, setRiskMaxPosition] = useState("20.0");
  const [riskMinConfidence, setRiskMinConfidence] = useState("0.0");
  const [riskCvarLimit, setRiskCvarLimit] = useState("10.0");
  const [kellyFraction, setKellyFraction] = useState("0.5");
  // Cycle 8: temporal-core architecture (lstm | tcn). Changing it requires a retrain.
  const [nnTrunk, setNnTrunk] = useState("lstm");

  const [installModalOpen, setInstallModalOpen] = useState(false);
  const [installState, setInstallState] = useState<any>({
    status: "idle", pct: 0, model: "", total_mb: 0, comp_mb: 0, speed_mb: 0, rem_time: 0, error_msg: "",
  });
  const [resettingPaper, setResettingPaper] = useState(false);

  // Which LLM credential fields are relevant to the selected provider.
  const needsGemini = provider === "gemini" || provider === "hybrid_gemini";
  const needsClaude = provider === "anthropic" || provider === "hybrid_claude";
  const needsOllama = provider.includes("ollama") || provider.includes("hybrid");

  const startPollingProgress = () => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/setup/ollama-progress?t=${Date.now()}`);
        if (res.ok) {
          const st = await res.json();
          setInstallState(st);
          if (st.status === "done" || st.status === "error") {
            clearInterval(interval);
            setTimeout(() => {
              if (st.status !== "error") alert(`Installation of ${st.model || "model"} completed successfully. Proceeding to Dashboard.`);
              window.location.href = "/dashboard";
            }, 1000);
          }
        }
      } catch (err) {
        // Backend restarted maybe? If so, we're likely done.
        clearInterval(interval);
        setTimeout(() => { window.location.href = "/dashboard"; }, 1500);
      }
    }, 500);
  };

  useEffect(() => {
    const fetchConfig = async (retries = 3) => {
      try {
        // reveal=1 returns the real (unredacted) .env values so the password-style
        // fields prefill with the actual keys (shown as dots, peekable via the eye).
        // Localhost, single-owner app — no admin key required.
        const res = await fetch(`http://127.0.0.1:8000/api/setup/config?reveal=1&t=${Date.now()}`);
        if (!res.ok) throw new Error("Backend not reachable");
        const cfg = await res.json();

        setProvider(cfg.AI_PROVIDER || "gemini");
        setOllamaModel(cfg.OLLAMA_MODEL || "llama3");

        // Keep obvious placeholders out of the fields, but keep any real value
        // (reveal=1 already returns the true secret).
        const isRealKey = (k: string) => !!(k && k.length > 3 && !k.toLowerCase().includes("your_") && !k.toLowerCase().includes("0x000") && !k.includes("*"));
        const sanitize = (k: string) => isRealKey(k) ? k : "";

        setGeminiKey(sanitize(cfg.GEMINI_API_KEY || ""));
        setAnthropicKey(sanitize(cfg.ANTHROPIC_API_KEY || ""));
        setPaperMode(cfg.PAPER_MODE === "true" || cfg.PAPER_MODE === true);

        // Live trading + strategy book
        const asBool = (v: any, d: boolean) => v === undefined || v === null ? d : (v === true || String(v).toLowerCase() === "true");
        setPaperTrading(asBool(cfg.PAPER_TRADING, true));
        setBinanceKey(sanitize(cfg.BINANCE_FUTURES_API_KEY || ""));
        setBinanceSecret(sanitize(cfg.BINANCE_FUTURES_SECRET || ""));
        setBinanceTestnet(asBool(cfg.BINANCE_FUTURES_TESTNET, true));
        if (cfg.BINANCE_FUTURES_LEVERAGE) setBinanceLeverage(String(cfg.BINANCE_FUTURES_LEVERAGE));
        setStrategyEnabled(asBool(cfg.STRATEGY_AGENT_ENABLED, false));
        if (cfg.STRATEGY_AGENT_SYMBOLS) setStrategySymbols(String(cfg.STRATEGY_AGENT_SYMBOLS));
        setTsSymbols(String(cfg.STRATEGY_AGENT_TS_SYMBOLS || ""));
        if (cfg.STRATEGY_AGENT_W_MANAGED) setWManaged(String(cfg.STRATEGY_AGENT_W_MANAGED));
        if (cfg.STRATEGY_AGENT_W_TS) setWTs(String(cfg.STRATEGY_AGENT_W_TS));
        setNewsOverlay(asBool(cfg.STRATEGY_AGENT_NEWS_OVERLAY, true));
        setNewsLlmVerify(asBool(cfg.STRATEGY_AGENT_NEWS_LLM_VERIFY, false));
        if (cfg.STRATEGY_AGENT_MAX_ORDER_USD) setMaxOrderUsd(String(cfg.STRATEGY_AGENT_MAX_ORDER_USD));

        setArbitrumRpcUrl(cfg.ARBITRUM_RPC_URL || "");
        setAgentPrivateKey(sanitize(cfg.AGENT_PRIVATE_KEY || ""));
        setAgentWalletAddress(sanitize(cfg.AGENT_WALLET_ADDRESS || ""));

        setWalletConnectProjectId(cfg.WALLETCONNECT_PROJECT_ID || cfg._extra?.WALLETCONNECT_PROJECT_ID || "");
        setRampHostApiKey(sanitize(cfg.RAMP_HOST_API_KEY || cfg._extra?.RAMP_HOST_API_KEY || ""));

        setAlpacaApiKey(sanitize(cfg.ALPACA_API_KEY || ""));
        setAlpacaSecretKey(sanitize(cfg.ALPACA_SECRET_KEY || ""));
        setXApiKey(sanitize(cfg.X_API_KEY || ""));
        setXApiSecret(sanitize(cfg.X_API_SECRET || ""));
        setXAccessToken(sanitize(cfg.X_ACCESS_TOKEN || ""));
        setXAccessTokenSecret(sanitize(cfg.X_ACCESS_TOKEN_SECRET || ""));
        setTelegramApiId(sanitize(cfg.TELEGRAM_API_ID || ""));
        setTelegramApiHash(sanitize(cfg.TELEGRAM_API_HASH || ""));

        if (cfg.RISK_MAX_DRAWDOWN_PCT) setRiskMaxDrawdown(String(cfg.RISK_MAX_DRAWDOWN_PCT));
        if (cfg.RISK_MAX_DAILY_LOSS_PCT) setRiskMaxDailyLoss(String(cfg.RISK_MAX_DAILY_LOSS_PCT));
        if (cfg.RISK_MAX_POSITION_PCT) setRiskMaxPosition(String(cfg.RISK_MAX_POSITION_PCT));
        if (cfg.RISK_MIN_CONFIDENCE !== undefined && cfg.RISK_MIN_CONFIDENCE !== null) setRiskMinConfidence(String(cfg.RISK_MIN_CONFIDENCE));
        if (cfg.RISK_CVAR_LIMIT_PCT) setRiskCvarLimit(String(cfg.RISK_CVAR_LIMIT_PCT));
        if (cfg.NN_KELLY_FRACTION !== undefined && cfg.NN_KELLY_FRACTION !== null) setKellyFraction(String(cfg.NN_KELLY_FRACTION));
        if (cfg.NN_TRUNK) setNnTrunk(String(cfg.NN_TRUNK).toLowerCase());

        setLoading(false);
      } catch (err) {
        if (retries > 0) {
          setTimeout(() => fetchConfig(retries - 1), 1000);
        } else {
          console.error("Failed to fetch setup config", err);
          setError("Couldn't load your saved configuration from the backend. Make sure it's running — the values below are defaults, so saving now may overwrite your .env.");
          setLoading(false);
        }
      }
    };

    fetchConfig();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setErrorDetails(null);

    try {
      const payload: Record<string, any> = {
        AI_PROVIDER: provider,
        OLLAMA_MODEL: needsOllama ? ollamaModel : undefined,
        GEMINI_API_KEY: geminiKey,
        ANTHROPIC_API_KEY: anthropicKey,
        PAPER_MODE: paperMode ? "true" : "false",
        ARBITRUM_RPC_URL: arbitrumRpcUrl,
        AGENT_PRIVATE_KEY: agentPrivateKey,
        AGENT_WALLET_ADDRESS: agentWalletAddress,
        WALLETCONNECT_PROJECT_ID: walletConnectProjectId,
        RAMP_HOST_API_KEY: rampHostApiKey,
        ALPACA_API_KEY: alpacaApiKey,
        ALPACA_SECRET_KEY: alpacaSecretKey,
        X_API_KEY: xApiKey,
        X_API_SECRET: xApiSecret,
        X_ACCESS_TOKEN: xAccessToken,
        X_ACCESS_TOKEN_SECRET: xAccessTokenSecret,
        TELEGRAM_API_ID: telegramApiId,
        TELEGRAM_API_HASH: telegramApiHash,
        RISK_MAX_DRAWDOWN_PCT: riskMaxDrawdown,
        RISK_MAX_DAILY_LOSS_PCT: riskMaxDailyLoss,
        RISK_MAX_POSITION_PCT: riskMaxPosition,
        RISK_MIN_CONFIDENCE: riskMinConfidence,
        RISK_CVAR_LIMIT_PCT: riskCvarLimit,
        NN_KELLY_FRACTION: kellyFraction,
        NN_TRUNK: nnTrunk,
        // Live trading + 2-sleeve strategy book
        PAPER_TRADING: paperTrading ? "true" : "false",
        BINANCE_FUTURES_API_KEY: binanceKey,
        BINANCE_FUTURES_SECRET: binanceSecret,
        BINANCE_FUTURES_TESTNET: binanceTestnet ? "true" : "false",
        BINANCE_FUTURES_LEVERAGE: binanceLeverage,
        STRATEGY_AGENT_ENABLED: strategyEnabled ? "true" : "false",
        STRATEGY_AGENT_SYMBOLS: strategySymbols,
        STRATEGY_AGENT_TS_SYMBOLS: tsSymbols,
        STRATEGY_AGENT_W_MANAGED: wManaged,
        STRATEGY_AGENT_W_TS: wTs,
        STRATEGY_AGENT_NEWS_OVERLAY: newsOverlay ? "true" : "false",
        STRATEGY_AGENT_NEWS_LLM_VERIFY: newsLlmVerify ? "true" : "false",
        STRATEGY_AGENT_MAX_ORDER_USD: maxOrderUsd,
      };

      // Never overwrite a stored secret with a blank field. Empty here means
      // "unchanged", so provider switches don't wipe unrelated credentials.
      for (const k of SECRET_KEYS) {
        if (!payload[k]) delete payload[k];
      }

      let successMsg = "Settings saved successfully. The backend will reboot with new keys.";
      try {
        const res = await fetch("http://127.0.0.1:8000/api/setup/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(`Server returned ${res.status}: ${errText}`);
        }

        const data = await res.json();

        if (data.installing_model) {
          setInstallModalOpen(true);
          startPollingProgress();
          return; // Wait for background download to finish before resolving settings page.
        }
        if (data.message) successMsg = data.message;
      } catch (fetchErr: any) {
        const msg = (fetchErr?.message || "").toLowerCase();
        // Browser network disconnects happen because saving .env auto-restarts the backend instantly.
        if (fetchErr instanceof TypeError || msg.includes("fetch") || msg.includes("network") || msg.includes("connection") || msg.includes("failed")) {
          console.warn("Backend .env reloaded the server. Assuming success.", fetchErr);
        } else {
          throw fetchErr;
        }
      }

      alert(successMsg);
      localStorage.removeItem("setupSkipped");
      window.location.href = "/dashboard";
    } catch (err: any) {
      setError("Network or server error while saving settings.");
      setErrorDetails(err.message || err.toString());
      setSaving(false);
    }
  };

  const handleResetPaper = async () => {
    if (!paperMode) return;
    if (!window.confirm("Reset paper trading? This will delete paper trades, clear paper statements, and restore the starting balance.")) return;
    setResettingPaper(true);
    setError(null);
    setErrorDetails(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/trading/reset-paper", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: true }),
      });
      if (!res.ok) throw new Error(await res.text());
      alert("Paper trading reset. Starting balance restored and trades cleared.");
      window.location.href = "/dashboard";
    } catch (err: any) {
      setError("Failed to reset paper trading.");
      setErrorDetails(err?.message || err.toString());
      setResettingPaper(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center bg-transparent text-zinc-500">
        <div className="flex items-center gap-3">
          <span className="h-2 w-2 animate-ping rounded-full bg-zinc-400" />
          <span className="animate-pulse text-sm tracking-wide">Initializing interface…</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen overflow-y-auto bg-transparent p-6 font-sans text-white md:p-10">
      <div className="relative mx-auto mt-2 max-w-2xl">
        {/* Decorative background flare */}
        <div className="pointer-events-none absolute -top-10 right-0 h-80 w-80 rounded-full bg-white/[0.03] blur-[120px]" />

        <Link href="/dashboard" className="mb-8 inline-flex items-center text-xs uppercase tracking-widest text-zinc-500 transition-colors hover:text-zinc-200">
          <ArrowLeft size={14} className="mr-2" /> Return to Dashboard
        </Link>

        <h1 className="mb-2 text-3xl font-semibold tracking-tight">System Configuration</h1>
        <p className="mb-10 max-w-lg text-sm leading-relaxed text-zinc-500">
          Manage your intelligence providers and execution credentials. Saving these parameters triggers an automatic core reboot.
        </p>

        {error && (
          <div className="mb-8 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-400">
            <div className="font-semibold">{error}</div>
            {errorDetails && (
              <details className="mt-2 cursor-pointer text-xs text-red-400/80 outline-none">
                <summary className="flex items-center font-semibold uppercase tracking-widest transition-colors hover:text-red-300">Show full error</summary>
                <pre className="mt-4 whitespace-pre-wrap rounded-lg border border-red-500/20 bg-black/40 p-3 font-mono text-[11px] leading-relaxed">{errorDetails}</pre>
              </details>
            )}
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-6">

          {/* ── AI Trading Engine ── */}
          <Section
            title="AI Trading Engine"
            dotColor="bg-zinc-600"
            info={
              <>
                <div className="mb-2 font-semibold text-zinc-100">AI Trading Engine</div>
                The AI interprets real-time news severity and extracts trade sentiment.<br /><br />
                <span className="text-zinc-400">Gemini / Claude:</span> highest accuracy, external API. Best for production.<br /><br />
                <span className="text-zinc-400">Ollama:</span> runs locally. Zero API cost, needs strong hardware.<br /><br />
                <span className="text-zinc-400">Hybrid:</span> local Ollama for routine work, escalates to Gemini/Claude for hard calls.
              </>
            }
          >
            <div className="space-y-5">
              <Field label="LLM Provider" htmlFor="provider">
                <SelectInput id="provider" value={provider} onChange={(e) => setProvider(e.target.value)}>
                  <option value="gemini">Gemini Only</option>
                  <option value="anthropic">Claude Only</option>
                  <option value="ollama">Ollama Only (Local)</option>
                  <option value="hybrid_gemini">Hybrid: Ollama + Gemini (Recommended)</option>
                  <option value="hybrid_claude">Hybrid: Ollama + Claude</option>
                </SelectInput>
              </Field>

              {needsOllama && (
                <Field
                  label="Ollama Model"
                  htmlFor="ollama-model"
                  hint="If the selected model isn't installed locally, the backend pulls it automatically."
                >
                  <SelectInput id="ollama-model" value={ollamaModel} onChange={(e) => setOllamaModel(e.target.value)}>
                    <option value="llama3">llama3 (Default - 8B)</option>
                    <option value="llama3.1">llama3.1 (Latest 8B)</option>
                    <option value="llama3.2">llama3.2 (Latest 3B - Fast)</option>
                    <option value="llama3:70b">llama3:70b (Requires 40GB+ VRAM)</option>
                    <option value="llama4">llama4 (Latest Generation)</option>
                    <option value="mistral">mistral (7B)</option>
                  </SelectInput>
                </Field>
              )}

              {needsGemini && (
                <Field label="Gemini API Key" htmlFor="gemini-key">
                  <SecretInput id="gemini-key" value={geminiKey} onChange={(e) => setGeminiKey(e.target.value)} placeholder="AIza…" />
                </Field>
              )}

              {needsClaude && (
                <Field label="Claude API Key (Anthropic)" htmlFor="anthropic-key">
                  <SecretInput id="anthropic-key" value={anthropicKey} onChange={(e) => setAnthropicKey(e.target.value)} placeholder="sk-ant-…" />
                </Field>
              )}
            </div>
          </Section>

          {/* ── Trading Mode (DeFi/Web3 execution) ── */}
          <Section
            title="Trading Mode"
            dotColor="bg-zinc-600"
            info={
              <>
                <div className="mb-2 font-semibold text-zinc-100">Trading Mode</div>
                Whether the on-chain (Arbitrum) engine places real swaps or simulates them.<br /><br />
                <span className="text-zinc-400">Paper:</span> virtual funds, no real risk.<br /><br />
                <span className="text-zinc-400">Live:</span> real capital on Arbitrum. Requires a funded agent wallet.
              </>
            }
          >
            <div className="space-y-4">
              <ControlRow
                title="Safe Test Mode (Paper)"
                description="Simulate on-chain trades without using real funds."
              >
                <Toggle checked={paperMode} onChange={() => setPaperMode(!paperMode)} />
              </ControlRow>
              {paperMode && (
                <button
                  type="button"
                  onClick={handleResetPaper}
                  disabled={resettingPaper}
                  className="inline-flex items-center rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-[12px] font-semibold uppercase tracking-widest text-amber-300 transition-colors hover:bg-amber-500/20 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {resettingPaper ? "Resetting…" : "Reset Paper Trading"}
                </button>
              )}
              {!paperMode && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-[12px] font-medium text-red-400">
                  WARNING: Live on-chain trading is active. Real swaps execute on Arbitrum. Ensure the agent wallet holds sufficient USDC and gas.
                </div>
              )}
            </div>
          </Section>

          {/* ── Live Trading & Strategy Book ── */}
          <Section title="Live Trading & Strategy Book" dotColor={paperTrading ? "bg-emerald-500" : "bg-red-500"}>
            <div className="space-y-5">
              <ControlRow
                title="Paper Trading (master switch)"
                description={<>ON = simulated, no real orders. OFF = route real orders <span className="text-zinc-400">where valid broker keys exist</span>.</>}
              >
                <Toggle checked={paperTrading} onChange={() => setPaperTrading(!paperTrading)} tone="safety" />
              </ControlRow>
              {!paperTrading && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-3 text-[12px] font-medium text-red-400">
                  LIVE TRADING ARMED. Real orders route on any venue with valid keys (Alpaca for ETFs/crypto-spot; Binance USD-M futures for perps). Each venue stays paper until its keys are present; Binance also needs Testnet OFF below.
                </div>
              )}

              <ControlRow
                title="Run the Strategy Book"
                description="Managed-beta + optional 4h TS-momentum sleeve, alongside the NN agent."
              >
                <Toggle checked={strategyEnabled} onChange={() => setStrategyEnabled(!strategyEnabled)} />
              </ControlRow>

              {strategyEnabled && (
                <div className="space-y-5">
                  <Field label="Managed-beta universe (daily)" htmlFor="strat-symbols">
                    <TextInput id="strat-symbols" mono value={strategySymbols} onChange={(e) => setStrategySymbols(e.target.value)} placeholder="SPY QQQ TQQQ TLT GLD BTC-USD ETH-USD" />
                  </Field>
                  <Field
                    label={<>TS-momentum sleeve (Binance perps, 4h) <span className="text-zinc-600">— blank = managed-only</span></>}
                    htmlFor="ts-symbols"
                  >
                    <TextInput id="ts-symbols" mono value={tsSymbols} onChange={(e) => setTsSymbols(e.target.value)} placeholder="BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT" />
                  </Field>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { label: "Managed weight", val: wManaged, set: setWManaged, ph: "0.5" },
                      { label: "TS weight", val: wTs, set: setWTs, ph: "0.5" },
                      { label: "Max order ($)", val: maxOrderUsd, set: setMaxOrderUsd, ph: "5000" },
                    ].map((f) => (
                      <Field key={f.label} label={<span className="text-[12px]">{f.label}</span>}>
                        <TextInput type="number" step="0.05" mono value={f.val} onChange={(e) => f.set(e.target.value)} placeholder={f.ph} className="px-3 py-2" />
                      </Field>
                    ))}
                  </div>
                  <ControlRow title="News risk overlay" description="Directional news can veto / de-gear / halt an asset.">
                    <Toggle checked={newsOverlay} onChange={() => setNewsOverlay(!newsOverlay)} />
                  </ControlRow>
                  <ControlRow title="LLM RiskAgent cross-check" description="Adds an LLM veto (tighten-only). Needs an LLM key.">
                    <Toggle checked={newsLlmVerify} onChange={() => setNewsLlmVerify(!newsLlmVerify)} />
                  </ControlRow>
                </div>
              )}

              <Divider className="my-1" />
              <div>
                <h4 className="text-sm font-medium text-zinc-300">
                  Binance USD-M Futures <span className="text-[12px] text-zinc-600">(perps + shorting — the TS sleeve venue)</span>
                </h4>
              </div>
              <Field label="API Key" htmlFor="binance-key">
                <SecretInput id="binance-key" value={binanceKey} onChange={(e) => setBinanceKey(e.target.value)} placeholder="Binance futures API key" />
              </Field>
              <Field label="Secret" htmlFor="binance-secret">
                <SecretInput id="binance-secret" value={binanceSecret} onChange={(e) => setBinanceSecret(e.target.value)} placeholder="Binance futures secret" />
              </Field>
              <div className="grid grid-cols-2 items-start gap-3">
                <ControlRow title="Testnet" description="ON = safe sandbox. OFF = mainnet.">
                  <Toggle checked={binanceTestnet} onChange={() => setBinanceTestnet(!binanceTestnet)} tone="safety" />
                </ControlRow>
                <Field label="Max leverage" htmlFor="binance-lev">
                  <TextInput id="binance-lev" type="number" step="1" mono value={binanceLeverage} onChange={(e) => setBinanceLeverage(e.target.value)} placeholder="2" />
                </Field>
              </div>
            </div>
          </Section>

          {/* ── Trading Wallets (Web3) ── */}
          <Section
            title="Trading Wallets (Web3)"
            dotColor="bg-zinc-600"
            info={
              <>
                <div className="mb-2 font-semibold text-zinc-100">Trading Wallets</div>
                Where your AI stores and uses its on-chain capital.<br /><br />
                <span className="text-zinc-400">RPC URL:</span> gateway node to the Arbitrum blockchain.<br /><br />
                <span className="text-zinc-400">Private Key:</span> the AI&apos;s signing key. Do not share.
              </>
            }
          >
            <div className="space-y-5">
              <Field label="Network Gateway URL (Arbitrum RPC)" htmlFor="rpc-url">
                <TextInput id="rpc-url" mono value={arbitrumRpcUrl} onChange={(e) => setArbitrumRpcUrl(e.target.value)} placeholder="https://arb-mainnet.g.alchemy.com/v2/…" />
              </Field>
              <Field label="Wallet Private Key (optional — needed for live on-chain trades)" htmlFor="agent-pk">
                <SecretInput id="agent-pk" value={agentPrivateKey} onChange={(e) => setAgentPrivateKey(e.target.value)} placeholder="0x…" />
              </Field>
              <Field label="Public Wallet Address" htmlFor="agent-addr">
                <TextInput id="agent-addr" mono value={agentWalletAddress} onChange={(e) => setAgentWalletAddress(e.target.value)} placeholder="0x…" />
              </Field>
              <Field label={<>WalletConnect Project ID <span className="text-zinc-600">(optional — enables the connect QR)</span></>} htmlFor="wc-id">
                <TextInput id="wc-id" mono value={walletConnectProjectId} onChange={(e) => setWalletConnectProjectId(e.target.value)} placeholder="from cloud.reown.com" />
              </Field>
              <Field label={<>Ramp Host API Key <span className="text-zinc-600">(optional — enables Add Funds / Google Pay)</span></>} htmlFor="ramp-key">
                <SecretInput id="ramp-key" value={rampHostApiKey} onChange={(e) => setRampHostApiKey(e.target.value)} placeholder="from ramp.network" />
              </Field>

              <WalletPanel />
            </div>
          </Section>

          {/* ── Data Sources & Social ── */}
          <Section
            title="Data Sources & Social"
            dotColor="bg-zinc-600"
            info={
              <>
                <div className="mb-2 font-semibold text-zinc-100">Data Sources</div>
                Lets the AI gather sentiment and pricing from external networks.<br /><br />
                <span className="text-zinc-400">Alpaca:</span> traditional stock pricing + brokerage.<br /><br />
                <span className="text-zinc-400">X (Twitter):</span> public sentiment and hype signals.
              </>
            }
          >
            <div className="space-y-5">
              <h4 className="text-sm font-medium text-zinc-300">Alpaca Markets</h4>
              <Field label="API Key" htmlFor="alpaca-key">
                <SecretInput id="alpaca-key" value={alpacaApiKey} onChange={(e) => setAlpacaApiKey(e.target.value)} placeholder="PK…" />
              </Field>
              <Field label="Secret Key" htmlFor="alpaca-secret">
                <SecretInput id="alpaca-secret" value={alpacaSecretKey} onChange={(e) => setAlpacaSecretKey(e.target.value)} placeholder="Secret…" />
              </Field>

              <Divider className="my-2" />

              <h4 className="text-sm font-medium text-zinc-300">X (Twitter) Integration</h4>
              <Field label="X API Key" htmlFor="x-key">
                <SecretInput id="x-key" value={xApiKey} onChange={(e) => setXApiKey(e.target.value)} placeholder="Key" />
              </Field>
              <Field label="X API Secret" htmlFor="x-secret">
                <SecretInput id="x-secret" value={xApiSecret} onChange={(e) => setXApiSecret(e.target.value)} placeholder="Secret" />
              </Field>
              <Field label="X Access Token" htmlFor="x-token">
                <SecretInput id="x-token" value={xAccessToken} onChange={(e) => setXAccessToken(e.target.value)} placeholder="Token" />
              </Field>
              <Field label="X Access Token Secret" htmlFor="x-token-secret">
                <SecretInput id="x-token-secret" value={xAccessTokenSecret} onChange={(e) => setXAccessTokenSecret(e.target.value)} placeholder="Secret" />
              </Field>

              <Divider className="my-2" />

              <h4 className="text-sm font-medium text-zinc-300">Telegram Integration</h4>
              <Field label="API ID" htmlFor="tg-id">
                <SecretInput id="tg-id" value={telegramApiId} onChange={(e) => setTelegramApiId(e.target.value)} placeholder="ID" />
              </Field>
              <Field label="API Hash" htmlFor="tg-hash">
                <SecretInput id="tg-hash" value={telegramApiHash} onChange={(e) => setTelegramApiHash(e.target.value)} placeholder="Hash" />
              </Field>
            </div>
          </Section>

          {/* ── Advanced Options — risk parameters ── */}
          <div className="rounded-2xl border border-[#171717] bg-[#0A0A0A] p-6 md:p-7">
            <button type="button" onClick={() => setAdvancedOpen(!advancedOpen)} className="flex w-full items-center justify-between text-left">
              <span className="text-[15px] font-semibold text-zinc-100">Advanced Options — Risk Parameters</span>
              <span className="text-sm text-zinc-400">{advancedOpen ? "Hide" : "Show"}</span>
            </button>

            {advancedOpen && (
              <div className="mt-6 grid grid-cols-1 gap-5 md:grid-cols-2">
                <p className="text-[12px] text-zinc-500 md:col-span-2">
                  These tune the live risk engine. Changes apply on the next agent restart (triggered by Save).
                </p>
                {[
                  { label: "Max Portfolio Drawdown (%)", val: riskMaxDrawdown, set: setRiskMaxDrawdown, hint: "Hard halt above this drawdown" },
                  { label: "Max Daily Loss (%)", val: riskMaxDailyLoss, set: setRiskMaxDailyLoss, hint: "Stops trading for the day past this loss" },
                  { label: "Max Single Position (%)", val: riskMaxPosition, set: setRiskMaxPosition, hint: "Cap on capital deployed per trade" },
                  { label: "Min Confidence (0-1)", val: riskMinConfidence, set: setRiskMinConfidence, hint: "Reject trades below this model confidence" },
                  { label: "CVaR Tail-Risk Limit (%)", val: riskCvarLimit, set: setRiskCvarLimit, hint: "Block new trades above this projected tail loss (0 disables)" },
                  { label: "Kelly Fraction (0 = use model size)", val: kellyFraction, set: setKellyFraction, hint: "Fraction of Kelly to bet; lower = more conservative" },
                ].map((f) => (
                  <Field key={f.label} label={f.label} hint={f.hint}>
                    <TextInput type="number" step="0.01" mono value={f.val} onChange={(e) => f.set(e.target.value)} />
                  </Field>
                ))}
                <Field
                  className="md:col-span-2"
                  label="Model Architecture (temporal core)"
                  hint={<>Switches the model&apos;s temporal core. You must <span className="text-zinc-400">retrain</span> with this setting and load that checkpoint — the live model cold-starts if the checkpoint&apos;s core doesn&apos;t match.</>}
                >
                  <SelectInput value={nnTrunk} onChange={(e) => setNnTrunk(e.target.value)}>
                    <option value="lstm">LSTM (default — recurrent)</option>
                    <option value="tcn">TCN (causal temporal conv-net)</option>
                  </SelectInput>
                </Field>
              </div>
            )}
          </div>

          <div className="flex justify-end pt-2 pb-12">
            <button
              type="submit"
              disabled={saving}
              className="flex items-center rounded-lg bg-zinc-100 px-7 py-3 font-semibold text-zinc-900 shadow-lg transition-all hover:bg-white disabled:opacity-50"
            >
              {saving ? "Saving…" : <><Save size={18} className="mr-2" /> Save &amp; Restart Agents</>}
            </button>
          </div>
        </form>
      </div>

      {/* Model Download Modal */}
      {installModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-md">
          <div className="relative w-full max-w-lg overflow-hidden rounded-2xl border border-[#171717] bg-[#0A0A0A] p-8 shadow-2xl">
            <div className="mb-8 flex flex-col items-center justify-center">
              <h2 className="mb-2 text-xl font-semibold">
                {installState.status === "installing_ollama" && installState.pct < 1
                  ? "Downloading Ollama Engine…"
                  : installState.status === "installing_ollama"
                  ? "Installing Ollama Engine…"
                  : installState.status === "starting_ollama"
                  ? "Starting Ollama Service…"
                  : installState.status === "pulling_model"
                  ? (installState.ollama_status && !installState.ollama_status.includes("pulling ") && !installState.ollama_status.includes("downloading ")
                     ? (installState.ollama_status.charAt(0).toUpperCase() + installState.ollama_status.slice(1) + "…")
                     : `Downloading ${installState.model}`)
                  : installState.status === "done"
                  ? "Installation Complete!"
                  : "Preparing Engine…"}
              </h2>
              <p className="text-center text-sm text-zinc-400">
                Please keep this window open while the background processor downloads external dependencies.
              </p>
            </div>

            {(installState.status === "pulling_model" || (installState.status === "installing_ollama" && installState.total_mb > 0)) && (
              <div className="animate-in fade-in zoom-in space-y-5 duration-300">
                <div className="relative mb-2 h-3 w-full overflow-hidden rounded-full border border-zinc-800 bg-zinc-800/60">
                  <div className="h-full rounded-full bg-zinc-100 transition-all duration-300 ease-out" style={{ width: `${(installState.pct * 100).toFixed(1)}%` }} />
                </div>
                <div className="grid grid-cols-2 gap-x-2 gap-y-4 text-sm">
                  <div className="flex flex-col pl-1">
                    <span className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">Total Progress</span>
                    <span className="font-mono text-zinc-100">{(installState.pct * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex flex-col pr-1 text-right">
                    <span className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">File Progress</span>
                    <span className="font-mono text-zinc-100">
                      {installState.comp_mb >= 1024 ? `${(installState.comp_mb / 1024).toFixed(2)} GB` : `${installState.comp_mb.toFixed(1)} MB`}
                      <span className="mx-1 text-xs text-zinc-500">/</span>
                      {installState.total_mb >= 1024 ? `${(installState.total_mb / 1024).toFixed(2)} GB` : `${installState.total_mb.toFixed(1)} MB`}
                    </span>
                  </div>
                  <div className="flex flex-col pl-1">
                    <span className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">Network Speed</span>
                    <span className="font-mono text-emerald-400">
                      {installState.speed_mb >= 1024 ? `${(installState.speed_mb / 1024).toFixed(2)} GB/s` : `${installState.speed_mb.toFixed(1)} MB/s`}
                    </span>
                  </div>
                  <div className="flex flex-col pr-1 text-right">
                    <span className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">Time Remaining</span>
                    <span className="font-mono text-zinc-100">
                      {(() => {
                        if (installState.status === "pulling_model" && installState.pct >= 1) return "Finalizing…";
                        if (installState.rem_time === undefined || installState.rem_time === null || isNaN(installState.rem_time) || installState.rem_time < 0) return "Calculating…";
                        if (installState.rem_time === 0 && installState.total_mb > 0) return "Almost Done…";
                        const hrs = Math.floor(installState.rem_time / 3600);
                        const mins = Math.floor((installState.rem_time % 3600) / 60);
                        const secs = Math.floor(installState.rem_time % 60);
                        if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
                        if (mins > 0) return `${mins}m ${secs}s`;
                        return `${secs}s`;
                      })()}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {installState.status === "error" && (
              <div className="rounded-lg border border-red-500/20 bg-red-900/10 p-4 text-center font-mono text-sm text-red-400">
                {installState.error_msg || "Unknown error occurred during setup."}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
