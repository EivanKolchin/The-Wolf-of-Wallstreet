export const API_BASE = "http://127.0.0.1:8000/api";
export const WS_BASE = "ws://127.0.0.1:8000/ws";

export async function fetchFromAPI(endpoint: string, options?: RequestInit) {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) throw new Error("API error");
    return res.json();
}

export function subscribeToLiveWs(
    onMessage: (topic: string, data: any) => void
) {
    const ws = new WebSocket(`${WS_BASE}/live`);
    ws.onmessage = (event) => {
        try {
            const parsed = JSON.parse(event.data);
            if (Object.keys(parsed).length === 1) {
                const topic = Object.keys(parsed)[0];
                onMessage(topic, parsed[topic]);
            }
        } catch (e) {
            console.error("WS parse error", e);
        }
    };
    return ws;
}
