const { spawn } = require('child_process');

async function main() {
  console.log('Starting db.exe...');
  const proc = spawn('.\\db.exe', [], { stdio: 'inherit' });

  proc.on('exit', (code, sig) => {
    console.log(`[db.exe exited with code: ${code}, signal: ${sig}]`);
  });

  await new Promise(r => setTimeout(r, 1200));

  try {
    const testSuite = require('./test_suite_fetch.js');
  } catch (e) {
    console.error('Test error:', e);
  }
}

main();
