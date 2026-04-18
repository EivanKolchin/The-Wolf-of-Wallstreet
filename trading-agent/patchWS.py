import re

with open(r'frontend/components/TradingChart.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the old WebSocket logically to connect to both @kline AND @aggTrade to get tick-by-tick streams

old_ws = '''        const wsUrl = wss://stream.binance.com:9443/ws/@kline_;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.e === 'kline') {
                const k = message.k;
                const newCandle = {
                    time: k.t / 1000,
                    open: parseFloat(k.o),
                    high: parseFloat(k.h),
                    low: parseFloat(k.l),
                    close: parseFloat(k.c),
                };

                if (seriesRef.current) {
                    seriesRef.current.update(newCandle as any);
                }'''

new_ws = '''        const safeSymbol = symbol.toLowerCase().replace(/[^a-z0-9]/g, '');
        const wsUrl = wss://stream.binance.com:9443/stream?streams=@kline_/@aggTrade;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const payload = JSON.parse(event.data);
            if (!payload.data) return;
            const message = payload.data;
            
            // Handle Tick-by-Tick trade data via aggTrade
            if (message.e === 'aggTrade') {
                if (!chartDataRef.current || chartDataRef.current.length === 0) return;
                const lastIdx = chartDataRef.current.length - 1;
                const lastCandle = chartDataRef.current[lastIdx];
                
                const tickPrice = parseFloat(message.p);
                
                // Update live real-time candle high/low/close bounds based on individual ticks
                const updatedCandle = {
                    ...lastCandle,
                    high: Math.max(lastCandle.high, tickPrice),
                    low: Math.min(lastCandle.low, tickPrice),
                    close: tickPrice,
                };
                
                chartDataRef.current[lastIdx] = updatedCandle;
                if (seriesRef.current) {
                    seriesRef.current.update(updatedCandle as any);
                }
                return;
            }

            // Handle historical base formation from kline (e.g. 1m interval boundary logic)
            if (message.e === 'kline') {
                const k = message.k;
                const newCandle = {
                    time: k.t / 1000,
                    open: parseFloat(k.o),
                    high: parseFloat(k.h),
                    low: parseFloat(k.l),
                    close: parseFloat(k.c),
                };

                if (seriesRef.current) {
                    seriesRef.current.update(newCandle as any);
                }'''

if old_ws in text:
    text = text.replace(old_ws, new_ws)
    print("Successfully patched")
else:
    print("Old match not found")

with open(r'frontend/components/TradingChart.tsx', 'w', encoding='utf-8') as f:
    f.write(text)
