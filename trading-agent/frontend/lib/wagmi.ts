import { http, createConfig } from "wagmi";
import { mainnet, sepolia } from "wagmi/chains";

// Specifically for Kite Chain, assuming generic testnet or custom chain configs if Kite AI isn't predefined.
const kiteChain = {
    id: 8453, // placeholder chain id for generic EVM
    name: 'Kite AI',
    nativeCurrency: { name: 'KITE', symbol: 'KITE', decimals: 18 },
    rpcUrls: {
        default: { http: [process.env.NEXT_PUBLIC_KITE_RPC as string || 'https://rpc.testnet.gokite.ai'] },
    },
    blockExplorers: {
        default: { name: 'Kitescan', url: 'https://testnet.kitescan.ai' },
    },
};

export const config = createConfig({
    chains: [kiteChain as any, mainnet],
    transports: {
        [kiteChain.id]: http(),
        [mainnet.id]: http(),
    },
});
