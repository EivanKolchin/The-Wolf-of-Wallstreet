"use client";

import { useEffect, useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { Copy, Check, CreditCard, QrCode } from "lucide-react";
import { API_BASE } from "@/lib/api";

// Workstream D/E: deposit-address QR + Google-Pay (Ramp) on-ramp, for Settings.
// Reads the publishable config from /api/wallet/config. Degrades gracefully:
// the deposit QR needs only the wallet address; the on-ramp needs a Ramp key.
export function WalletPanel() {
  const [cfg, setCfg] = useState<any>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/wallet/config`).then((r) => r.json()).then(setCfg).catch(() => {});
  }, []);

  const addr: string = cfg?.deposit_address || "";
  const uri: string = cfg?.deposit_uri || addr;
  const onramp = cfg?.onramp || {};
  const onrampEnabled = !!onramp?.enabled;

  const copy = () => {
    if (!addr) return;
    navigator.clipboard?.writeText(addr);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const openOnramp = async () => {
    if (!onrampEnabled || !addr) return;
    try {
      const mod: any = await import("@ramp-network/ramp-instant-sdk");
      const RampInstantSDK = mod.RampInstantSDK || mod.default?.RampInstantSDK || mod.default;
      new RampInstantSDK({
        hostAppName: "Wolf of Wallstreet",
        hostLogoUrl: "https://ramp.network/assets/images/Ramp-Logo.svg",
        hostApiKey: onramp.ramp_host_api_key,
        userAddress: addr,
        swapAsset: cfg?.onramp_asset || "ARBITRUM_USDC",
        variant: "auto",
      }).show();
    } catch (e) {
      console.error("Ramp widget failed to open", e);
    }
  };

  return (
    <div className="rounded-2xl border border-neutral-800/60 bg-[#0C0C0E] p-5">
      <div className="flex items-center gap-2 mb-4">
        <QrCode size={16} className="text-zinc-400" />
        <h3 className="text-[13px] font-semibold uppercase tracking-widest text-neutral-300">
          Fund the Agent Wallet
        </h3>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {/* Deposit QR */}
        <div className="flex flex-col items-center text-center">
          {addr ? (
            <>
              <div className="rounded-xl bg-[#0A0A0A] border border-zinc-800 p-3">
                <QRCodeSVG value={uri} size={150} bgColor="#0A0A0A" fgColor="#e4e4e7" level="M" />
              </div>
              <p className="text-[11px] text-zinc-500 mt-2">
                Scan to send <span className="text-zinc-300">USDC on Arbitrum</span> to the agent.
              </p>
              <button
                onClick={copy}
                className="mt-2 flex items-center gap-1.5 text-[11px] font-mono text-zinc-400 hover:text-zinc-200 transition-colors"
                title="Copy address"
              >
                {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                {addr.slice(0, 8)}…{addr.slice(-6)}
              </button>
            </>
          ) : (
            <p className="text-xs text-zinc-500 max-w-[220px]">
              Set your <span className="text-zinc-300">Public Wallet Address</span> above and save to
              show a scannable deposit QR here.
            </p>
          )}
        </div>

        {/* On-ramp */}
        <div className="flex flex-col justify-center gap-3">
          <p className="text-xs text-zinc-400 leading-relaxed">
            Buy crypto with <span className="text-zinc-200">Google Pay</span>, a card, or a bank
            transfer — delivered straight to the agent wallet on Arbitrum.
          </p>
          <button
            onClick={openOnramp}
            disabled={!onrampEnabled || !addr}
            className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
          >
            <CreditCard size={16} />
            Add Funds
          </button>
          {!onrampEnabled && (
            <p className="text-[11px] text-zinc-600">
              Add a <span className="text-zinc-400">{(onramp?.provider || "ramp")} key</span> in the
              keys section to enable funding.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
