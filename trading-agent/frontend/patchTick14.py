import pathlib
import sys
file_path = pathlib.Path('components/TradingChart.tsx')
text = file_path.read_text(encoding='utf-8')
lines = text.splitlines()
for i, line in enumerate(lines):
    if 'const wsUrl =' in line:
        lines[i] = '        const wsUrl = wss://stream.binance.com:9443/stream?streams=@kline_/@aggTrade;'
        break
file_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
