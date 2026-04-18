import pathlib
file_path = pathlib.Path('components/TradingChart.tsx')
text = file_path.read_text(encoding='utf-8')
import re
new_text = re.sub(r'const wsUrl = wss://stream\.binance\.com.*?@aggTrade;', 'const wsUrl = "\\wss://stream.binance.com:9443/stream?streams=\\$\\{safeSymbol\\}@kline_\\$\\{fetchTimeframe.toLowerCase()\\}/\\$\\{safeSymbol\\}@aggTrade\\";', text)
new_text = new_text.replace('"\\', '').replace('\\"', '').replace('\\$', '$').replace('\\{', '{').replace('\\}', '}')
file_path.write_text(new_text, encoding='utf-8')
