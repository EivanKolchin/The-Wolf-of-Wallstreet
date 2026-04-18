import pathlib
file_path = pathlib.Path('components/TradingChart.tsx')
text = file_path.read_text(encoding='utf-8')
text = text.replace('const wsUrl = wss://stream.binance.com:9443/stream?streams=@kline_/@aggTrade;', 'const wsUrl = wss://stream.binance.com:9443/stream?streams=@kline_/@aggTrade;')
file_path.write_text(text, encoding='utf-8')
