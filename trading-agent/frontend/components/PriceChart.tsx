import React from 'react';
import { AdvancedRealTimeChart } from 'react-ts-tradingview-widgets';

export default function PriceChart({ data }: { data?: any[] }) {
  return (
    <div className="w-full h-full min-h-[400px]">
      <AdvancedRealTimeChart
        symbol="BINANCE:BTCUSD"
        theme="dark"
        autosize
        allow_symbol_change={false}
        calendar={false}
        studies={["MASimple@tv-basicstudies", "Auto Fib Retracement", "EMA@tv-basicstudies"]}
        style="1"
        toolbar_bg="#000000"
        backgroundColor="#000000"
        
        hide_top_toolbar={false}
        hide_legend={false}
        save_image={false}
        container_id="tradingview_chart"
        interval="15"
        theme="dark"
        
        timezone="Etc/UTC"
      />
    </div>
  );
}
