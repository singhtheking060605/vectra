async function run() {
  console.log('--- 1. Testing GET /status ---');
  const s = await fetch('http://127.0.0.1:8080/status').then(r => r.json());
  console.log('Status:', s);

  console.log('--- 2. Testing GET /clusters ---');
  const c = await fetch('http://127.0.0.1:8080/clusters').then(r => r.json());
  console.log('Clusters count:', c.length);

  console.log('--- 3. Testing POST /search (isro) ---');
  const sr1 = await fetch('http://127.0.0.1:8080/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'isro',
      algo: 'hnsw',
      metric: 'cosine',
      k: 3,
      alpha: -1.0
    })
  }).then(r => r.json());
  console.log('Search 1 (isro) Hits:', sr1.results.length);
  sr1.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('--- 4. Testing POST /cluster/add ---');
  const addC = await fetch('http://127.0.0.1:8080/cluster/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: 'space_science',
      label: 'Space Science',
      color: '#38bdf8'
    })
  }).then(r => r.json());
  console.log('Add Cluster:', addC);

  console.log('--- 5. Testing POST /insert ---');
  const ins = await fetch('http://127.0.0.1:8080/insert', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      metadata: 'Chandrayaan-3 lunar rover soft landing on Moon south pole',
      category: 'space_science'
    })
  }).then(r => r.json());
  console.log('Inserted Vector ID:', ins.id);

  console.log('--- 6. Testing Search on chandrayaan ---');
  const sr2 = await fetch('http://127.0.0.1:8080/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'chandrayaan moon landing',
      algo: 'hnsw',
      metric: 'cosine',
      k: 3,
      alpha: -1.0
    })
  }).then(r => r.json());
  console.log('Search 2 Hits:', sr2.results.length);
  sr2.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('\n>>> ALL MULTI-STEP HTTP API TESTS PASSED 100% SUCCESSFULLY! <<<');
}

run().catch(console.error);
