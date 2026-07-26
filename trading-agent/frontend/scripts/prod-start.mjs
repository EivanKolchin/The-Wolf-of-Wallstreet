// Launcher used by start.bat / start.sh: serve the production build for fast
// page loads, rebuilding only when the source changed since the last build.
// If the build fails for any reason, fall back to dev mode so the app still runs.
import { spawnSync, spawn } from 'node:child_process';
import { existsSync, statSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import net from 'node:net';
import http from 'node:http';
import path from 'node:path';

const root = process.cwd();
const nextBin = path.join(root, 'node_modules', 'next', 'dist', 'bin', 'next');
const buildIdFile = path.join(root, '.next', 'BUILD_ID');
// Sidecar recording the source mtime that produced the current build. We compare against
// THIS (not BUILD_ID's own mtime, which Next/tooling can touch independently) so a stale
// .next never looks "up to date".
const srcStampFile = path.join(root, '.next', '.prod-src-mtime');
const PORT = Number(process.env.PORT) || 3000;

const SOURCE_DIRS = ['app', 'components', 'lib'];
const SOURCE_FILES = ['package.json', 'next.config.mjs', 'tailwind.config.ts', 'tsconfig.json'];
const SKIP = new Set(['node_modules', '.next', '.git']);

function newestMtime(p) {
  let newest = 0;
  let st;
  try { st = statSync(p); } catch { return 0; }
  if (st.isFile()) return st.mtimeMs;
  if (!st.isDirectory()) return 0;
  for (const entry of readdirSync(p)) {
    if (SKIP.has(entry)) continue;
    newest = Math.max(newest, newestMtime(path.join(p, entry)));
  }
  return newest;
}

function sourceMtime() {
  let newest = 0;
  for (const d of SOURCE_DIRS) newest = Math.max(newest, newestMtime(path.join(root, d)));
  for (const f of SOURCE_FILES) newest = Math.max(newest, newestMtime(path.join(root, f)));
  return newest;
}

function runNext(args) {
  return spawnSync(process.execPath, ['--max-old-space-size=8192', nextBin, ...args], {
    stdio: 'inherit',
    env: process.env,
  });
}

// True if something is already listening on the port.
function portInUse(port) {
  return new Promise((resolve) => {
    const socket = net.connect({ host: '127.0.0.1', port }, () => { socket.destroy(); resolve(true); });
    socket.on('error', () => resolve(false));
    socket.setTimeout(1500, () => { socket.destroy(); resolve(false); });
  });
}

// True if the port answers an HTTP request (i.e. the app is already up, not a zombie socket).
function portServesHttp(port) {
  return new Promise((resolve) => {
    const req = http.get({ host: '127.0.0.1', port, path: '/', timeout: 2500 }, (res) => {
      res.destroy(); resolve(true);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

function readStamp() {
  try { return Number(readFileSync(srcStampFile, 'utf8').trim()) || 0; } catch { return 0; }
}

function startFreshOrDev() {
  const src = sourceMtime();
  // Stale unless there's a real build (BUILD_ID) whose recorded source-stamp is >= current source.
  const stale = !existsSync(buildIdFile) || src > readStamp();

  if (stale) {
    console.log('[prod-start] Source changed since last build (or no build found) — building...');
    const res = runNext(['build']);
    if (res.status !== 0) {
      console.error('[prod-start] Build FAILED — falling back to dev mode so the app still runs.');
      const dev = spawn(process.execPath, ['--max-old-space-size=8192', nextBin, 'dev', '-p', String(PORT)], {
        stdio: 'inherit',
        env: process.env,
      });
      dev.on('exit', (code) => process.exit(code ?? 1));
      process.on('SIGINT', () => dev.kill('SIGINT'));
      return;
    }
    // Record the source mtime that this build reflects (re-read after build in case it changed).
    try { writeFileSync(srcStampFile, String(sourceMtime())); } catch { /* non-fatal */ }
    console.log('[prod-start] Build OK — starting production server.');
  } else {
    console.log('[prod-start] Existing build is up to date — starting production server.');
  }
  process.exit(runNext(['start', '-p', String(PORT)]).status ?? 0);
}

async function main() {
  if (await portInUse(PORT)) {
    // Something already holds the port. Don't crash-loop into the launcher's
    // "npm install" fallback (a port conflict is not a dependency problem) —
    // exit(0) so the launcher stops here with a clear, actionable message.
    if (await portServesHttp(PORT)) {
      console.log(`[prod-start] A frontend is ALREADY running at http://localhost:${PORT} — reusing it, not starting a second server.`);
      console.log('[prod-start] If you meant to run the new production build, close that window/server first, then relaunch.');
    } else {
      console.error(`[prod-start] Port ${PORT} is in use but not responding (a stale/zombie process is holding it).`);
      console.error(`[prod-start] Free it, then relaunch. Windows: for /f "tokens=5" %a in ('netstat -ano ^| findstr :${PORT}') do taskkill /F /PID %a`);
      console.error(`[prod-start] macOS/Linux: lsof -ti:${PORT} | xargs kill -9`);
    }
    process.exit(0);
  }
  startFreshOrDev();
}

main();
