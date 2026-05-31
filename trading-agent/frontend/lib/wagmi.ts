import { http, createConfig, type Config } from "wagmi";
import { mainnet, arbitrum } from "wagmi/chains";
import { injected, walletConnect } from "wagmi/connectors";

// Execution venue for the crypto agent is Arbitrum One (Uniswap V3).
//
// Workstream D: build the wagmi config with explicit connectors.
//  - injected()      → MetaMask (and other browser-extension wallets).
//  - walletConnect() → shows a QR to scan with a mobile wallet. Requires a free
//    projectId (cloud.reown.com), fetched at runtime from /api/wallet/config so
//    it lives in the backend .env with no frontend rebuild. Omitted when absent.
export function buildConfig(projectId?: string): Config {
    const connectors = projectId
        ? [injected({ shimDisconnect: true }), walletConnect({ projectId, showQrModal: true })]
        : [injected({ shimDisconnect: true })];
    return createConfig({
        chains: [arbitrum, mainnet],
        connectors,
        transports: {
            [arbitrum.id]: http(),
            [mainnet.id]: http(),
        },
    });
}

// Default (injected-only) config for first paint / fallback. Providers upgrades
// this to include WalletConnect once the projectId loads.
export const config = buildConfig();
