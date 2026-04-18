import re

with open(r'frontend/components/TradingChart.tsx', 'r', encoding='utf-8') as f:
    text = f.read()

# We need to change the websocket connection from stream.binance.com to accept trades/aggTrades AND klines to reconstruct the tick-by-tick data streaming. 
# Alternatively, since TradingView lightweight-charts Candlestick series can be updated with the latest price, we can subscribe to @aggTrade or @trade and update the current candlestick's close/high/low in real-time.

# Let's see the current websocket connection
match = re.search(r'(const wsUrl = wss://stream\.binance\.com:9443/ws/\$\{symbol\.toLowerCase\(\)\.replace\(/\[\^a-z0-9\]/g, \'\'\)\}.*?;)', text, re.DOTALL)
if match:
    print("Found WebSocket URL:", match.group(1))
else:
    print("WebSocket URL not found.")

