"use client";

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WagmiProvider, type Config } from 'wagmi';
import { buildConfig, config as defaultConfig } from '../lib/wagmi';
import { AppStateProvider } from '../lib/context';
import { useEffect, useState } from 'react';
import { API_BASE } from '../lib/api';

const queryClient = new QueryClient();

export function Providers({ children }: { children: React.ReactNode }) {
  // Workstream D: render immediately with the injected-only config, then upgrade
  // to include the WalletConnect connector once the projectId loads from the
  // backend. The user isn't connected on first paint, so the one-time swap is
  // harmless and avoids blocking render on the fetch.
  const [config, setConfig] = useState<Config>(defaultConfig);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/wallet/config`)
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        const pid: string = d?.walletconnect_project_id || "";
        if (pid) setConfig(buildConfig(pid));
      })
      .catch(() => { /* keep injected-only config */ });
    return () => { cancelled = true; };
  }, []);

  return (
    <WagmiProvider config={config} reconnectOnMount>
      <QueryClientProvider client={queryClient}>
        <AppStateProvider>
          {children}
        </AppStateProvider>
      </QueryClientProvider>
    </WagmiProvider>
  );
}
