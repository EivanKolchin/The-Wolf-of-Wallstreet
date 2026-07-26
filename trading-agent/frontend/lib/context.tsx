"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { API_BASE, subscribeToLiveWs } from "./api";
import { PortfolioStatus, Trade, NewsImpact } from "./types";

interface AppState {
    status: { running: boolean; paper_mode: boolean; address: string };
    portfolio: PortfolioStatus | null;
    signals: any[];
    positions: Trade[];
    news: NewsImpact | null;
    currency: string;
    exchangeRates: Record<string, number>;
    setCurrency: (currency: string) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: React.ReactNode }) {
    const [state, setState] = useState<Omit<AppState, 'setCurrency'>>({
        status: { running: true, paper_mode: true, address: "0xAgentPlaceholder..." },
        portfolio: null,
        signals: [],
        positions: [],
        news: null,
        currency: "USD",
        exchangeRates: { USD: 1 },
    });

    useEffect(() => {
        // Fetch exchange rates from an open API initially
        fetch("https://api.exchangerate-api.com/v4/latest/USD")
            .then(res => res.json())
            .then(data => {
                if (data && data.rates) {
                    setState(prev => ({ ...prev, exchangeRates: data.rates }));
                }
            })
            .catch(err => console.error("Failed to fetch exchange rates", err));

        const ws = subscribeToLiveWs((topic, data) => {
            setState((prev) => {
                if (topic === "state") {
                    return { ...prev, status: { ...prev.status, ...data } };
                }
                if (topic === "portfolio") {
                    return { ...prev, portfolio: data };
                }
                if (topic === "signal") {
                    return { ...prev, signals: [data, ...prev.signals].slice(0, 20) };
                }
                if (topic === "trade") {
                    // Closed trades leave the positions list; open ones upsert.
                    const rest = prev.positions.filter(p => p.id !== data.id);
                    return { ...prev, positions: data?.status === "closed" ? rest : [data, ...rest] };
                }
                if (topic === "news") {
                    return { ...prev, news: data };
                }
                return prev;
            });
        });

        // fetch initial from API
        fetch(`${API_BASE}/setup/config`)
            .then((res) => res.json())
            .then((data) => {
                setState((prev) => ({
                    ...prev,
                    status: {
                        ...prev.status,
                        paper_mode: data?.PAPER_MODE === "true"
                    }
                }));
            })
            .catch(console.error);

        // Seed open positions from the DB so the positions page shows existing
        // (paper) trades on load instead of waiting for the next WS trade event.
        const seedPositions = () =>
            fetch(`${API_BASE}/positions`)
                .then((res) => res.json())
                .then((rows) => {
                    if (Array.isArray(rows)) {
                        setState((prev) => ({ ...prev, positions: rows }));
                    }
                })
                .catch(console.error);
        seedPositions();
        const positionsPoll = setInterval(seedPositions, 30000);

        return () => { clearInterval(positionsPoll); ws.close(); };
    }, []);

    return (
        <AppContext.Provider value={{
            ...state,
            setCurrency: (c: string) => setState(prev => ({ ...prev, currency: c }))
        }}>
            {children}
        </AppContext.Provider>
    );    
}

export function useAppState() {
    const ctx = useContext(AppContext);
    if (!ctx) throw new Error("useAppState must be used inside AppStateProvider");
    return ctx;
}
