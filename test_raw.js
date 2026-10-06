const http = require('http');

function post(path, data) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(data);
    const req = http.request({
      hostname: '127.0.0.1',
      port: 8080,
      path: path,
      method: 'POST',
      agent: false,
      headers: {
        'Connection': 'close',
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload)
      }
    }, res => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => {
        try { resolve(JSON.parse(b)); } catch(e) { resolve(b); }
      });
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

function get(path) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: '127.0.0.1',
      port: 8080,
      path: path,
      method: 'GET',
      agent: false,
      headers: { 'Connection': 'close' }
    }, res => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => {
        try { resolve(JSON.parse(b)); } catch(e) { resolve(b); }
      });
    });
    req.on('error', reject);
    req.end();
  });
}

async function run() {
  console.log('Testing /status...');
  const st = await get('/status');
  console.log('Status:', st);

  console.log('Testing /cluster/add...');
  const cRes = await post('/cluster/add', { name: 'finance', label: 'Finance', color: '#10b981' });
  console.log('Cluster Add:', cRes);

  console.log('Testing /insert...');
  const iRes = await post('/insert', { metadata: 'Stock Market equities', category: 'finance' });
  console.log('Insert:', iRes);

  console.log('Testing /search...');
  const sRes = await post('/search', { query: 'stock market', k: 3, alpha: 1.0 });
  console.log('Search hits:', sRes.results ? sRes.results.length : 0);
  console.log('SUCCESS!');
}

run().catch(console.error);
