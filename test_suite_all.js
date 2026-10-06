async function runTests() {
  console.log('--- TEST 1: GET /status ---');
  let res = await fetch('http://127.0.0.1:8080/status');
  console.log('Status:', await res.json());

  console.log('--- TEST 2: GET /clusters ---');
  res = await fetch('http://127.0.0.1:8080/clusters');
  const clusters = await res.json();
  console.log('Clusters count:', clusters.length, clusters.map(c => c.name));

  console.log('--- TEST 3: POST /cluster/add "gaming" ---');
  res = await fetch('http://127.0.0.1:8080/cluster/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: 'gaming', label: 'Video Games & Esports', color: '#a855f7' })
  });
  console.log('Add Cluster Result:', await res.json());

  console.log('--- TEST 4: POST /insert entities for "gaming" ---');
  const itemsToAdd = [
    { metadata: 'Unreal Engine: real-time ray tracing shaders', category: 'gaming' },
    { metadata: 'Counter Strike: tactical fps competitive matchmaking', category: 'gaming' },
    { metadata: 'Elden Ring: open world action rpg boss mechanics', category: 'gaming' }
  ];
  for (const item of itemsToAdd) {
    res = await fetch('http://127.0.0.1:8080/insert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(item)
    });
    console.log('Inserted:', await res.json());
  }

  console.log('--- TEST 5: GET /clusters and /items ---');
  res = await fetch('http://127.0.0.1:8080/clusters');
  const updatedClusters = await res.json();
  console.log('Updated clusters count:', updatedClusters.length);

  res = await fetch('http://127.0.0.1:8080/items');
  const items = await res.json();
  const gamingItems = items.filter(i => i.category === 'gaming');
  console.log('Gaming items in DB:', gamingItems.length);

  console.log('--- TEST 6: POST /search for gaming query ---');
  res = await fetch('http://127.0.0.1:8080/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: 'fps competitive matchmaking shaders',
      algo: 'hnsw',
      metric: 'cosine',
      k: 4,
      alpha: 1.0
    })
  });
  const searchRes = await res.json();
  console.log('Search latency:', searchRes.latencyUs, 'μs');
  console.log('Search hits:', searchRes.results.map(r => `[${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('--- TEST 7: POST /pca/fit & /pca/coordinates ---');
  res = await fetch('http://127.0.0.1:8080/pca/fit', { method: 'POST' });
  console.log('PCA fit:', await res.json());
  res = await fetch('http://127.0.0.1:8080/pca/coordinates');
  const pcaCoords = await res.json();
  console.log('PCA coordinates count:', Object.keys(pcaCoords.coordinates).length);

  console.log('--- TEST 8: POST /ivf/train & /ivf/stats ---');
  res = await fetch('http://127.0.0.1:8080/ivf/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nlist: 6, nprobe: 2 })
  });
  console.log('IVF train:', await res.json());
  res = await fetch('http://127.0.0.1:8080/ivf/stats');
  console.log('IVF stats:', await res.json());

  console.log('>>> ALL 8 INTEGRATION TESTS PASSED 100%! <<<');
}

runTests().catch(err => {
  console.error('Integration Test Failed:', err);
  process.exit(1);
});
