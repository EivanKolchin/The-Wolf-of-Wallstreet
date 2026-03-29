import { useState, useEffect } from "react";
import { API_BASE } from "../api";

export function useNewsData() {
    const [rawNews, setRawNews] = useState<any[]>([]);
    const [predictions, setPredictions] = useState<any[]>([]);

    useEffect(() => {
        let isMounted = true;
        
        async function fetchNews() {
            try {
                const [rawRes, insightsRes] = await Promise.all([
                    fetch(`${API_BASE}/news/raw`),
                    fetch(`${API_BASE}/news/recent`)
                ]);
                
                if (isMounted) {
                    if (rawRes.ok) setRawNews(await rawRes.json());
                    if (insightsRes.ok) setPredictions(await insightsRes.json());
                }
            } catch (err) {
                console.error("Failed to fetch news data:", err);
            }
        }

        fetchNews();
        const interval = setInterval(fetchNews, 5000);
        
        return () => {
            isMounted = false;
            clearInterval(interval);
        };
    }, []);

    return { rawNews, predictions };
}