"use client";

import { useConnect } from "wagmi";
import { X, Wallet, QrCode } from "lucide-react";

// Workstream D: connect modal offering MetaMask (injected) + WalletConnect (QR).
// The WalletConnect connector pops its own QR modal on connect (showQrModal),
// so scanning with mobile MetaMask "just works".
export function WalletConnectModal({ onClose }: { onClose: () => void }) {
  const { connectors, connect, isPending } = useConnect();

  const iconFor = (name: string) =>
    name.toLowerCase().includes("walletconnect") ? <QrCode size={18} /> : <Wallet size={18} />;

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-xl p-4"
      onClick={onClose}
    >
      <div
        className="bg-[#0C0C0E] border border-neutral-800/60 rounded-2xl max-w-sm w-full p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-base font-semibold text-zinc-100">Connect a wallet</h3>
          <button onClick={onClose} className="text-zinc-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-col gap-2">
          {connectors.map((c) => (
            <button
              key={c.uid}
              disabled={isPending}
              onClick={() => { connect({ connector: c }); onClose(); }}
              className="flex items-center gap-3 px-4 py-3 rounded-xl border border-zinc-800 bg-[#121214] text-zinc-200 hover:bg-zinc-800/60 hover:border-zinc-700 transition-all disabled:opacity-50"
            >
              {iconFor(c.name)}
              <span className="text-sm font-medium">{c.name}</span>
            </button>
          ))}
          {connectors.length === 0 && (
            <p className="text-xs text-zinc-500">No wallet connectors available.</p>
          )}
        </div>

        <p className="text-[11px] text-zinc-600 mt-4 leading-relaxed">
          <span className="text-zinc-400">MetaMask</span> uses your browser extension.
          {" "}<span className="text-zinc-400">WalletConnect</span> shows a QR code to scan
          with a mobile wallet. (WalletConnect needs a projectId in Settings.)
        </p>
      </div>
    </div>
  );
}
