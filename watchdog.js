// Vectra Server Keep-Alive Daemon
// Restarts db.exe automatically if it crashes
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

const VECTRA_DIR = path.dirname(__filename);
const PORT = 8081;
const MINGW = 'C:\\Users\\ARSHSH~1\\AppData\\Local\\Microsoft\\WinGet\\Packages\\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\\mingw64\\bin';

let serverProc = null;
let restartCount = 0;

function startServer() {
  if (serverProc) {
    try { serverProc.kill(); } catch(e) {}
    serverProc = null;
  }

  restartCount++;
  console.log(`[WATCHDOG] Starting db.exe (attempt #${restartCount})...`);

  const env = Object.assign({}, process.env);
  env.PATH = MINGW + ';' + env.PATH;

  serverProc = spawn('db.exe', [String(PORT)], {
    cwd: VECTRA_DIR,
    env: env,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  serverProc.stdout.on('data', d => process.stdout.write('[SERVER] ' + d));
  serverProc.stderr.on('data', d => process.stderr.write('[SERVER ERR] ' + d));

  serverProc.on('exit', (code, signal) => {
    console.log(`[WATCHDOG] db.exe exited (code=${code}, signal=${signal}). Restarting in 2s...`);
    serverProc = null;
    setTimeout(startServer, 2000);
  });

  serverProc.on('error', (err) => {
    console.error('[WATCHDOG] Failed to start:', err.message);
    setTimeout(startServer, 3000);
  });
}

// Ping check every 10 seconds
function healthCheck() {
  const req = http.get({ hostname: '127.0.0.1', port: PORT, path: '/status', timeout: 2000 }, (res) => {
    // alive
  });
  req.on('error', () => {
    console.log('[WATCHDOG] Health check failed - server may be down');
    if (!serverProc) startServer();
  });
  req.end();
}

console.log('[WATCHDOG] Vectra Keep-Alive daemon starting...');
startServer();
setInterval(healthCheck, 10000);

process.on('SIGINT', () => {
  console.log('\n[WATCHDOG] Shutting down...');
  if (serverProc) serverProc.kill();
  process.exit(0);
});
