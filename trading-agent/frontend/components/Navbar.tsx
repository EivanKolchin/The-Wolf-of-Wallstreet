"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAccount, useConnect, useDisconnect } from "wagmi";
import { useAppState } from "../lib/context";
import { Settings, AlertCircle } from "lucide-react";
import { ClockPanel } from "./ClockPanel";
import { useEffect, useState } from "react";

export default function Navbar() {
  const { status } = useAppState();
  const { address, isConnected } = useAccount();
  const { connectors, connect } = useConnect();
  const { disconnect } = useDisconnect();
  const pathname = usePathname();

  const [missingKeys, setMissingKeys] = useState(false);

  useEffect(() => {
    fetch("http://localhost:8000/api/setup/status")
      .then(res => res.json())
      .then(data => {
        if (data && data.needs_setup) setMissingKeys(true);
      })
      .catch(() => setMissingKeys(true));
  }, []);

  const navItems = [
    { name: "Overview", path: "/dashboard" },
    { name: "Positions", path: "/positions" },
    { name: "Intelligence", path: "/news" },
    { name: "Audit Log", path: "/audit" },
  ];

  return (
    <nav className="flex items-center justify-between px-8 py-4 border-b border-[#171717] bg-[#000000]/80 backdrop-blur-2xl sticky top-0 z-50 transition-all font-sans">
      
      {/* Brand & Main Nav */}
      <div className="flex items-center space-x-10">
        <Link href="/dashboard" className="text-[15px] font-semibold tracking-tight text-zinc-100 hover:opacity-80 transition-opacity flex items-center gap-2">        
          <div className="w-3 h-3 rounded-[3px] bg-zinc-600"></div>
          W.O.W.
        </Link>

        <div className="hidden md:flex space-x-6">
          {navItems.map((item) => (
            <Link
              key={item.name} 
              href={item.path}
              className={`text-[13px] tracking-wide transition-colors ${        
                pathname === item.path
                  ? "text-zinc-100 font-medium"
                  : "text-zinc-400 hover:text-zinc-100"
              }`}
            >
              {item.name}
            </Link>
          ))}
        </div>
      </div>

      {/* Utilities & States */}
      <div className="flex items-center space-x-5">

        <ClockPanel />

        {/* Missing Keys Setup Alert */}
        {missingKeys && (
          <Link href="/settings" className="flex items-center space-x-2.5 px-6 py-2 rounded-md bg-[#C45A3E]/5 hover:bg-[#C45A3E]/10 border border-[#C45A3E]/20 text-[#C45A3E] transition-all cursor-pointer">
            <AlertCircle size={16} />
            <span className="text-[13px] font-medium uppercase tracking-widest">Setup Required</span>
          </Link>
        )}

        {/* Wallet Connection */}
        {isConnected ? (
          <div className="flex items-center space-x-3">
            <span className="text-[12px] text-zinc-500 font-mono tracking-wider">
              {address?.slice(0, 6)}...{address?.slice(-4)}
            </span>
            <button
              onClick={() => disconnect()}
              className="text-[11px] uppercase tracking-widest text-zinc-400 hover:text-zinc-100 transition-colors"
            >
              Disconnect
            </button>
          </div>
        ) : (
          <button
            onClick={() => connect({ connector: connectors[0] })}
            className="text-[13px] font-medium tracking-wide text-zinc-300 bg-transparent hover:text-white border border-[#27272A] hover:border-[#3F3F46] hover:bg-[#18181B] px-8 py-2 rounded-md transition-all"   
          >
            Connect
          </button>
        )}

        <div className="w-[1px] h-4 bg-[#0A0A0A]"></div>

        {/* Settings Icon */}
        <Link href="/settings" className="text-zinc-400 hover:text-zinc-100 transition-colors p-1">
          <Settings size={18} strokeWidth={1.5} />
        </Link>
      </div>
    </nav>
  );
}
