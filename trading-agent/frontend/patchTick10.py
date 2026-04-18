import pathlib
file_path = pathlib.Path('components/TradingChart.tsx')
text = file_path.read_text(encoding='utf-8')
import re
text = re.sub(r'wss://stream\.binance\.com:9443/stream\?streams=@kline_/@aggTrade', r'wss://stream.binance.com:9443/stream?streams=@kline_/@aggTrade', text)
file_path.write_text(text, encoding='utf-8')
