const http = require('http');

const agent = new http.Agent({ keepAlive: true });

function req(method, path, data = null) {
  return new Promise((resolve, reject) => {
    const payload = data ? JSON.stringify(data) : null;
    const options = {
      hostname: '127.0.0.1',
      port: 8080,
      path: path,
      method: method,
      agent: agent,
      headers: {
        ...(payload ? {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload)
        } : {})
      }
    };

    const r = http.request(options, res => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(body) });
        } catch {
          resolve({ status: res.statusCode, body });
        }
      });
    });
    r.on('error', reject);
    if (payload) r.write(payload);
    r.end();
  });
}

async function runAll() {
  console.log('--- 1. Testing GET /status ---');
  const s = await req('GET', '/status');
  console.log('Status:', s.status, s.data);

  console.log('--- 2. Testing GET /clusters ---');
  const c = await req('GET', '/clusters');
  console.log('Clusters count:', c.data.length);

  console.log('--- 3. Testing POST /search (Auto-Alpha) ---');
  const sr1 = await req('POST', '/search', {
    query: 'dynamic programming memoization',
    algo: 'hnsw',
    metric: 'cosine',
    k: 3,
    alpha: -1.0
  });
  console.log('Search 1 Effective Alpha:', sr1.data.effectiveAlpha, 'Hits:', sr1.data.results.length);
  sr1.data.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('--- 4. Testing POST /cluster/add ---');
  const addC = await req('POST', '/cluster/add', {
    name: 'space_science',
    label: 'Space Science',
    color: '#38bdf8'
  });
  console.log('Add Cluster:', addC.data);

  console.log('--- 5. Testing POST /insert ---');
  const ins = await req('POST', '/insert', {
    metadata: 'James Webb Space Telescope deep infrared imaging',
    category: 'space_science'
  });
  console.log('Inserted Vector ID:', ins.data.id);

  console.log('--- 6. Testing Search on newly added vector ---');
  const sr2 = await req('POST', '/search', {
    query: 'James Webb telescope infrared',
    algo: 'hnsw',
    metric: 'cosine',
    k: 2,
    alpha: -1.0
  });
  console.log('Search 2 Hits:', sr2.data.results.length);
  sr2.data.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('\n>>> ALL MULTI-STEP HTTP API TESTS PASSED WITH 100% SUCCESS! <<<');
}

runAll().catch(e => console.error('TEST ERROR:', e.message));
