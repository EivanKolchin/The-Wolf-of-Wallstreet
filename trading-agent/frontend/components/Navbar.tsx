"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAccount, useConnect, useDisconnect } from "wagmi";
import { useAppState } from "../lib/context";
import { Settings, AlertCircle, Cpu, DollarSign, PoundSterling, Euro, ChevronDown } from "lucide-react";
import { ClockPanel } from "./ClockPanel";
import { useEffect, useState, useRef } from "react";
import { API_BASE } from "../api";

export default function Navbar() {
  const { status, currency, setCurrency } = useAppState();
  const { address, isConnected } = useAccount();
  const { connectors, connect } = useConnect();
  const { disconnect } = useDisconnect();
  const pathname = usePathname();

  const [missingKeys, setMissingKeys] = useState(false);
  const [missingList, setMissingList] = useState<any[]>([]);
  const [mounted, setMounted] = useState(false);
  const [dismissedBanner, setDismissedBanner] = useState(false);
  
  // Currency Dropdown State
  const [showCurrencyDrop, setShowCurrencyDrop] = useState(false);
  const currencyDropRef = useRef<HTMLDivElement>(null);

  // LLM Switch State
  const [llmStatus, setLlmStatus] = useState<any>(null);
  const [showLlmDrop, setShowLlmDrop] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
    fetch("http://127.0.0.1:8000/api/setup/status")
      .then(res => res.json())
      .then(data => {
        if (data && data.needs_setup) setMissingKeys(true);
        if (data && data.missing_integrations) setMissingList(data.missing_integrations);
      })
      .catch(() => setMissingKeys(true));

    const pollLLM = setInterval(() => {
      fetch("http://127.0.0.1:8000/api/llm/status")
        .then(res => res.json())
        .then(data => setLlmStatus(data))
        .catch(() => setLlmStatus(null));
    }, 2000);
    
    const handleOutsideClick = (e: MouseEvent) => {
      if (currencyDropRef.current && !currencyDropRef.current.contains(e.target as Node)) {
        setShowCurrencyDrop(false);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);

    return () => {
      clearInterval(pollLLM);
      document.removeEventListener("mousedown", handleOutsideClick);
    };
  }, []);

  const handleForceRevert = async () => {
    try {
      await fetch("http://127.0.0.1:8000/api/llm/force-revert", { method: 'POST' });
      setShowLlmDrop(false);
    } catch(e) {}
  };

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
          WoW
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

        {/* Currency Switcher */}
        <div className="relative" ref={currencyDropRef}>
          <button 
            onClick={() => setShowCurrencyDrop(!showCurrencyDrop)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-zinc-800 bg-[#121214] text-zinc-300 hover:bg-zinc-800 transition-colors"
          >
            <div className="flex items-center -space-x-1">
              <DollarSign size={14} className="text-zinc-400 z-10 bg-[#121214] rounded-full" />
              <PoundSterling size={14} className="text-zinc-600" />
            </div>
            <span className="text-[12px] font-medium tracking-wide">{currency}</span>
            <ChevronDown size={14} className={`text-zinc-500 transition-transform ${showCurrencyDrop ? 'rotate-180' : ''}`} />
          </button>

          {showCurrencyDrop && (
            <div className="absolute top-full mt-2 w-32 right-0 bg-[#121214] border border-[#27272a] rounded-xl shadow-2xl z-[999] p-1 flex flex-col fadeIn">
              {['USD', 'GBP', 'EUR', 'JPY', 'CAD'].map((code) => (
                <button 
                  key={code}
                  onClick={() => {
                    setCurrency(code);
                    setShowCurrencyDrop(false);
                  }}
                  className={`text-left px-3 py-2 rounded-lg text-xs transition-colors flex items-center justify-between ${
                    currency === code 
                      ? 'bg-zinc-800/80 text-zinc-200 font-medium' 
                      : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-300'
                  }`}
                >
                  {code}
                  {code === 'USD' && <DollarSign size={12} />}
                  {code === 'GBP' && <PoundSterling size={12} />}
                  {code === 'EUR' && <Euro size={12} />}
                  {code === 'JPY' && <span className="text-[12px]">¥</span>}
                  {code === 'CAD' && <DollarSign size={12} />}
                </button>
              ))}
            </div>
          )}
        </div>

        {status?.paper_mode && (
          <Link href="/settings" className="flex items-center space-x-2 px-3 py-1 rounded bg-[#FACC15]/10 border border-[#FACC15]/20 text-[#FACC15] transition-all cursor-pointer">
            <span className="text-[11px] font-medium uppercase tracking-widest">Paper Mode</span>
          </Link>
        )}

        <ClockPanel />

        {/* Informational Banner for missing non-critical dependencies */}
        {mounted && !missingKeys && (status as any)?.missing_integrations && (status as any).missing_integrations.length > 0 && !dismissedBanner && (
          <div className="group relative hidden lg:flex items-center space-x-2 px-4 py-1.5 rounded bg-orange-500/10 border border-orange-500/20 text-orange-400 transition-all cursor-help">
            <AlertCircle size={14} />
            <span className="text-[11px] font-medium uppercase tracking-widest">
              {(status as any).missing_integrations.map((i: any) => i.service).join(', ')} api missing
            </span>
            <button 
              onClick={(e) => {
                e.preventDefault();
                setDismissedBanner(true);
              }}
              className="ml-2 text-[9px] px-2 py-0.5 bg-orange-500/20 hover:bg-orange-500/40 rounded transition-colors uppercase tracking-widest"
            >
              Ignore for now
            </button>
            
            {/* Tooltip Hover for missing integrations */}
            <div className="absolute top-12 right-0 w-80 bg-[#161616] border border-[#333336] rounded-xl p-4 shadow-2xl opacity-0 group-hover:opacity-100 pointer-events-none group-hover:pointer-events-auto transition-opacity z-50">
              <h3 className="text-[12px] font-semibold text-neutral-100 uppercase tracking-widest mb-3 border-b border-[#333336] pb-2">Missing Data Sources</h3>
              <ul className="space-y-3">
                {(status as any).missing_integrations.map((item: any, idx: number) => (
                  <li key={idx} className="text-[11px] leading-relaxed">
                    <span className="text-orange-400 font-semibold block mb-0.5">{item.service}</span>
                    <span className="text-neutral-400">{item.impact}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-4 pt-2 border-t border-[#333336]">
                <Link href="/settings" className="text-[11px] text-blue-400 hover:text-blue-300 transition-colors uppercase tracking-widest font-semibold flex items-center">
                  Configure Settings &rarr;
                </Link>
              </div>
            </div>
          </div>
        )}

        {/* Missing Keys Setup Alert (Critical) */}
        {mounted && missingKeys && (
          <Link href="/settings" className="flex items-center space-x-2.5 px-6 py-2 rounded-md bg-[#C45A3E]/5 hover:bg-[#C45A3E]/10 border border-[#C45A3E]/20 text-[#C45A3E] transition-all cursor-pointer">
            <AlertCircle size={16} />
            <span className="text-[13px] font-medium uppercase tracking-widest">Setup Required</span>
          </Link>
        )}

        {/* LLM Status Dropdown */}
        {mounted && llmStatus?.is_overloaded && (
          <div className="relative" ref={dropRef}>
            <button 
              onClick={() => setShowLlmDrop(!showLlmDrop)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-orange-500/30 bg-orange-500/10 text-orange-400 hover:bg-orange-500/20 transition-all cursor-pointer shadow-lg animate-pulse"
            >
              <Cpu size={14} />
              <span className="text-[11px] uppercase tracking-widest font-mono">
                {llmStatus.time_remaining}s Cooldown
              </span>
            </button>

            {/* Dropdown Menu */}
            {showLlmDrop && (
              <div className="absolute top-10 right-0 w-[280px] bg-[#111] border border-[#2A2A2A] rounded-xl p-4 shadow-2xl z-50">
                <div className="flex items-center gap-2 mb-3 border-b border-[#2A2A2A] pb-3">
                  <AlertCircle size={16} className="text-orange-400" />
                  <span className="text-[12px] font-semibold text-zinc-200 uppercase tracking-wider">Engine Downgraded</span>
                </div>
                
                <p className="text-[11px] text-zinc-400 leading-relaxed mb-4">
                  The primary <span className="text-emerald-400">{llmStatus.primary_model}</span> crashed from extreme memory load. 
                  Currently running <span className="text-orange-400">{llmStatus.current_model}</span> to maintain feed processing.
                </p>

                <button 
                  onClick={handleForceRevert}
                  className="w-full py-2 bg-[#2A2A2A] hover:bg-[#3A3A3A] hover:text-white text-[11px] uppercase tracking-widest text-zinc-300 font-semibold rounded transition"
                >
                  Force Revert Now &rarr;
                </button>
              </div>
            )}
          </div>
        )}

        {/* Wallet Connection */}
        {mounted ? (
          isConnected ? (
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
          )
        ) : (
          <div className="w-[100px] h-9"></div>
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
