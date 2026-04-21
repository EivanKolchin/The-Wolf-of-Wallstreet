"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { subscribeToLiveWs } from "./api";
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
                    return { ...prev, positions: [data, ...prev.positions.filter(p => p.id !== data.id)] };
                }
                if (topic === "news") {
                    return { ...prev, news: data };
                }
                return prev;
            });
        });

        // fetch initial from API
        fetch("http://127.0.0.1:8000/api/setup/config")
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

        return () => ws.close();
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
