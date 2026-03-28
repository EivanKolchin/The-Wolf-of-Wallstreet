import re

with open('components/TradingChart.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Inject chartDataRef if not already there
ref_old = r'const prevRangeInfo = useRef<any>\(null\);'
ref_new = 'const prevRangeInfo = useRef<any>(null);\n    const chartDataRef = useRef<any[]>([]);'
if "const chartDataRef = useRef" not in text:
    text = re.sub(ref_old, ref_new, text)

# 2. Update chartDataRef in Fetch Data useEffect
fetch_sync_old = r'const uniqueData = formatted\.filter\(\(v: any, i: number, a: any\[\]\) => a\.findIndex\(\(t: any\) => \(t\.time === v\.time\)\) === i\);\n\s*setChartData\(uniqueData\);'
fetch_sync_new = 'const uniqueData = formatted.filter((v: any, i: number, a: any[]) => a.findIndex((t: any) => (t.time === v.time)) === i);\n                chartDataRef.current = uniqueData;\n                setChartData(uniqueData);'
text = re.sub(fetch_sync_old, fetch_sync_new, text)

# 3. Handle it inside loadMoreHistory
append_sync_old = r'setChartData\(prev => \{\n\s*const combined = \[\.\.\.formatted, \.\.\.prev\];\n\s*return combined\.filter\(\(v: any, i: number, a: any\[\]\) => a\.findIndex\(\(t: any\) => \(t\.time === v\.time\)\) === i\)\.sort\(\(a: any, b: any\) => a\.time - b\.time\);\n\s*\}\);'
append_sync_new = '''setChartData(prev => {
                        const combined = [...formatted, ...prev];
                        const unique = combined.filter((v: any, i: number, a: any[]) => a.findIndex((t: any) => (t.time === v.time)) === i).sort((a: any, b: any) => a.time - b.time);
                        chartDataRef.current = unique;
                        return unique;
                    });'''
text = re.sub(append_sync_old, append_sync_new, text)


# 4. Define `calculateEMA`
calc_ema_block = """    const calculateEMA = (data: any[], period: number) => {
        if (!data || data.length === 0) return [];
        const k = 2 / (period + 1);
        let emaData = [];
        let ema = data[0].close;
        for (let i = 0; i < data.length; i++) {
            ema = (data[i].close - ema) * k + ema;
            emaData.push({ time: data[i].time, value: ema });
        }
        return emaData;
    };"""

# Remove old calculateEMA if it exists
text = re.sub(r'\s*const calculateEMA = \([^)]+\) => \{.*?\n\s*\};\s*', '\n', text, flags=re.DOTALL)

# Insert it above Initialize Chart
init_chart_comment = '// Initialize Chart'
text = text.replace(init_chart_comment, calc_ema_block + '\n\n    ' + init_chart_comment)


# 5. Insert WebSocket code
ws_code = """    // WebSocket Live Updates
    useEffect(() => {
        if (cursorDate || !symbol) return; // Do not live update historical views
        let fetchTimeframe = timeframe;
        if (timeframe === '3M' || timeframe === '1Y' || timeframe === 'ALL') fetchTimeframe = '1M';
        
        const wsUrl = `wss://stream.binance.com:9443/ws/${symbol.toLowerCase().replace(/[^a-z0-9]/g, '')}@kline_${fetchTimeframe.toLowerCase()}`;
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
                    seriesRef.current.update(newCandle);
                }

                if (chartDataRef.current && chartDataRef.current.length > 0) {
                    const lastIdx = chartDataRef.current.length - 1;
                    const lastCandle = chartDataRef.current[lastIdx];
                    if (newCandle.time === lastCandle.time) {
                        chartDataRef.current[lastIdx] = newCandle;
                    } else if (newCandle.time > lastCandle.time) {
                        chartDataRef.current.push(newCandle);
                    }

                    // Dynamically recalculate EMA tails
                    if (config.ema9.show && ema9Ref.current) {
                         const emaTails = calculateEMA(chartDataRef.current, 9);
                         if (emaTails.length > 0) ema9Ref.current.update(emaTails[emaTails.length - 1]);
                    }
                    if (config.ema21.show && ema21Ref.current) {
                         const emaTails = calculateEMA(chartDataRef.current, 21);
                         if (emaTails.length > 0) ema21Ref.current.update(emaTails[emaTails.length - 1]);
                    }
                }
            }
        };

        return () => ws.close();
    }, [symbol, timeframe, cursorDate, config.ema9.show, config.ema21.show]);"""

fetch_marker = '// Fetch Data'
if "WebSocket Live Updates" not in text:
    text = text.replace(fetch_marker, ws_code + '\n\n    ' + fetch_marker)

# Update the initializer to handle stripped calculateEMA cleanly without throwing errors if EMA references it
# actually `calculateEMA` is cleanly global in this component due to previous step so no issues.

with open('components/TradingChart.tsx', 'w', encoding='utf-8') as f:
    f.write(text)
print('Live data integration injected')
