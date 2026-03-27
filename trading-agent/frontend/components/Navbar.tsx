"use client";

import Link from "next/link";
import { useAccount, useConnect, useDisconnect } from "wagmi";
import { useAppState } from "../lib/context";
import { Button } from "./ui/button";

export default function Navbar() {
  const { status } = useAppState();
  const { address, isConnected } = useAccount();
  const { connectors, connect } = useConnect();
  const { disconnect } = useDisconnect();

  return (
    <nav className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-950/50 backdrop-blur-md sticky top-0 z-50">
      <div className="flex items-center space-x-8">
        <Link href="/dashboard" className="text-xl font-bold tracking-tight text-white hover:text-zinc-200 transition-colors">
          Wolf of WallStreet <span className="text-blue-500">AI</span>
        </Link>
        <div className="hidden md:flex space-x-1">
          <Link href="/dashboard" className="text-sm font-medium text-zinc-400 hover:text-white px-3 py-2 rounded-md hover:bg-zinc-800/50 transition-colors">
            Dashboard
          </Link>
          <Link href="/positions" className="text-sm font-medium text-zinc-400 hover:text-white px-3 py-2 rounded-md hover:bg-zinc-800/50 transition-colors">
            Positions
          </Link>
          <Link href="/news" className="text-sm font-medium text-zinc-400 hover:text-white px-3 py-2 rounded-md hover:bg-zinc-800/50 transition-colors">
            News
          </Link>
          <Link href="/audit" className="text-sm font-medium text-zinc-400 hover:text-white px-3 py-2 rounded-md hover:bg-zinc-800/50 transition-colors">
            Audit
          </Link>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* Agent Status */}
        <div className="flex items-center space-x-2 bg-zinc-900 px-3 py-1.5 rounded-full border border-zinc-800">
          <div
            className={`w-2 h-2 rounded-full ${
              status.running ? "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]" : "bg-red-500"
            }`}
          />
          <span className="text-xs font-medium text-zinc-300">
            {status.running ? "Agent Active" : "Halted"}
          </span>
        </div>

        {/* Paper Mode Badge */}
        {status.paper_mode && (
          <div className="bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 px-3 py-1.5 rounded-full text-xs font-bold tracking-wider">
            PAPER MODE
          </div>
        )}

        {/* Wallet Connection */}
        {isConnected ? (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-zinc-400 font-mono bg-zinc-900 px-3 py-2 rounded-md border border-zinc-800">
              {address?.slice(0, 6)}...{address?.slice(-4)}
            </span>
            <Button variant="outline" size="sm" onClick={() => disconnect()} className="h-9">
              Disconnect
            </Button>
          </div>
        ) : (
          <Button
            size="sm"
            onClick={() => connect({ connector: connectors[0] })}
            className="h-9 font-medium"
          >
            Connect Wallet
          </Button>
        )}
      </div>
    </nav>
  );
}