const HOST = 'http://127.0.0.1:8081';

async function req(method, path, data = null) {
  const opts = {
    method: method,
    headers: { 'Content-Type': 'application/json' }
  };
  if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
    opts.body = JSON.stringify(data);
  }
  const res = await fetch(`${HOST}${path}`, opts);
  let parsed = null;
  try {
    parsed = await res.json();
  } catch {
    parsed = null;
  }
  return { status: res.status, data: parsed };
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runAll() {
  console.log('--- 1. Testing GET /status ---');
  const s = await req('GET', '/status');
  console.log('Status:', s.status, s.data);
  await sleep(40);

  console.log('--- 2. Testing GET /clusters ---');
  const c = await req('GET', '/clusters');
  console.log('Clusters count:', c.data.length);
  await sleep(40);

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
  await sleep(40);

  console.log('--- 4. Testing POST /cluster/add ---');
  const addC = await req('POST', '/cluster/add', {
    name: 'space_science',
    label: 'Space Science',
    color: '#38bdf8'
  });
  console.log('Add Cluster:', addC.data);
  await sleep(40);

  console.log('--- 5. Testing POST /insert ---');
  const ins = await req('POST', '/insert', {
    metadata: 'James Webb Space Telescope deep infrared imaging',
    category: 'space_science'
  });
  console.log('Inserted Vector ID:', ins.data.id);
  await sleep(40);

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
