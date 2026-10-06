const http = require('http');

const agent = new http.Agent({ keepAlive: true });

function search(query) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      query: query,
      algo: 'hnsw',
      metric: 'cosine',
      k: 3,
      alpha: -1.0
    });

    const req = http.request({
      hostname: '127.0.0.1',
      port: 8080,
      path: '/search',
      method: 'POST',
      agent: agent,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    }, res => {
      let body = '';
      res.on('data', d => body += d);
      res.on('end', () => resolve(JSON.parse(body)));
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function run() {
  console.log('Testing "isro" query:');
  const res = await search('isro');
  console.log('Results count:', res.results.length);
  res.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));

  console.log('\nTesting "chandrayaan" query:');
  const res2 = await search('chandrayaan');
  console.log('Results count:', res2.results.length);
  res2.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));
}

run().catch(console.error);
