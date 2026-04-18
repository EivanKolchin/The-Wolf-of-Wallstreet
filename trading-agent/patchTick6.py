import pathlib

file_path = pathlib.Path('frontend/components/TradingChart.tsx')
text = file_path.read_text(encoding='utf-8')

new_code = '''const safeSymbol = symbol.toLowerCase().replace(/[^a-z0-9]/g, '');
        const wsUrl = wss://stream.binance.com:9443/stream?streams=@kline_/@aggTrade;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const payload = JSON.parse(event.data);
            if (!payload?.data) return;
            const message = payload.data;

            if (message.e === 'aggTrade') {
                if (!chartDataRef.current || chartDataRef.current.length === 0) return;
                const lastIdx = chartDataRef.current.length - 1;
                const lastCandle = chartDataRef.current[lastIdx];
                const tickPrice = parseFloat(message.p);
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
                if (config.ema9.show && ema9Ref.current) {
                     const emaTails = calculateEMA(chartDataRef.current, 9);
                     if (emaTails.length > 0) ema9Ref.current.update(emaTails[emaTails.length - 1] as any);
                }
                if (config.ema21.show && ema21Ref.current) {
                     const emaTails = calculateEMA(chartDataRef.current, 21);
                     if (emaTails.length > 0) ema21Ref.current.update(emaTails[emaTails.length - 1] as any);
                }
                return;
            }

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

start_str = "        const wsUrl = \n"
end_str = "if (seriesRef.current) {\n                    seriesRef.current.update(newCandle as any);\n                }"

import re
# Escape everything from start_str out to end_str
combined = r"const wsUrl =[^;]+;.*?if \(seriesRef\.current\) \{\s+seriesRef\.current\.update\(newCandle as any\);\s+\}"

new_text = re.sub(combined, new_code, text, flags=re.DOTALL)
print(f"Matched: {new_text != text}")

file_path.write_text(new_text, encoding='utf-8')
