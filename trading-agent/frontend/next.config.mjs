/** @type {import('next').NextConfig} */
const nextConfig = {
    // NOTE: no `output: 'standalone'`. Standalone rewrites the server chunk layout for
    // containerized deploys and is incompatible with `next start` (breaks the viem vendor
    // chunk). The launcher runs `next start` locally, so the default build is what we want.
    // Allow an isolated build dir (e.g. NEXT_DIST_DIR=.next-prod) so a production build
    // can be produced/served without a concurrently-running `next dev` clobbering `.next`.
    distDir: process.env.NEXT_DIST_DIR || '.next',
    eslint: { ignoreDuringBuilds: false },
    typescript: { ignoreBuildErrors: false },
    webpack: (config) => {
        // wagmi / walletconnect / metamask-sdk reference optional deps that aren't used in the
        // browser build (a pino pretty-printer and a React-Native storage shim). Without this they
        // emit noisy "Module not found" warnings on every compile. Externalize / alias them away.
        config.externals.push('pino-pretty', 'lokijs', 'encoding');
        config.resolve.alias = {
            ...config.resolve.alias,
            '@react-native-async-storage/async-storage': false,
        };
        // Webpack's persistent filesystem cache gzip-serializes to disk. On this large bundle,
        // under memory pressure the Gzip/Gunzip buffer allocation fails
        // ("RangeError: Array buffer allocation failed") and CRASHES the dev server mid-navigation.
        // Disabling cache compression removes the failing (de)compression path entirely — the
        // cache still works (rebuilds stay fast), the on-disk files are just larger.
        if (config.cache && typeof config.cache === 'object') {
            config.cache.compression = false;
        }
        return config;
    },
};
export default nextConfig;
