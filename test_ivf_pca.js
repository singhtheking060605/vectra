const http = require('http');

function req(path, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const payload = data ? JSON.stringify(data) : null;
    const r = http.request({
      hostname: '127.0.0.1',
      port: 8080,
      path: path,
      method: method,
      agent: false,
      headers: {
        'Connection': 'close',
        ...(payload ? {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload)
        } : {})
      }
    }, res => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => resolve(JSON.parse(body)));
    });
    r.on('error', reject);
    if (payload) r.write(payload);
    r.end();
  });
}

async function runTests() {
  console.log('=== 1. IVF K-MEANS STATS ===');
  const ivfStats = await req('/ivf/stats');
  console.log(ivfStats);

  console.log('\n=== 2. IVF SEARCH: "delicious pizza" ===');
  const ivfSearch = await req('/search', 'POST', {
    query: 'delicious pizza',
    algo: 'ivf',
    metric: 'cosine',
    k: 4,
    alpha: -1.0
  });
  console.log(`Latency: ${ivfSearch.latencyUs}us, Hits:`);
  ivfSearch.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('\n=== 3. PCA 2D EIGENVECTOR COORDINATES (SAMPLE) ===');
  const pcaCoords = await req('/pca/coordinates');
  console.log('PCA Coordinates (First 5 items):', Object.entries(pcaCoords.coordinates).slice(0, 5));

  console.log('\n=== 4. PCA VECTOR PROJECTION ===');
  const pcaProj = await req('/pca/project', 'POST', { query: 'space planet cosmos telescope' });
  console.log('Projected 2D coordinate for space query:', pcaProj);

  console.log('\n=== 5. 4-WAY BENCHMARK BATTLE ===');
  const bench = await req('/bench?query=database+algorithm');
  console.log(`Dataset size: ${bench.vectorCount} vectors`);
  console.log(`  Brute-Force: ${bench.bruteForceUs} us`);
  console.log(`  KD-Tree:     ${bench.kdTreeUs} us`);
  console.log(`  HNSW Graph:  ${bench.hnswUs} us`);
  console.log(`  IVF K-Means: ${bench.ivfUs} us`);
}

runTests();
