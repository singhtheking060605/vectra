const http = require('http');

function post(path, data) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(data);
    const r = http.request({
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
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => resolve(JSON.parse(body)));
    });
    r.on('error', reject);
    r.write(payload);
    r.end();
  });
}

async function testCreateCluster() {
  console.log('1. Adding cluster "finance"...');
  const res1 = await post('/cluster/add', {
    name: 'finance',
    label: 'Finance & Markets',
    color: '#10b981'
  });
  console.log('Cluster Add Result:', res1);

  console.log('2. Adding entities into "finance"...');
  await post('/insert', { metadata: 'Stock Market: equities, dividend yields and index funds', category: 'finance' });
  await post('/insert', { metadata: 'Cryptocurrency: bitcoin, ethereum and blockchain consensus', category: 'finance' });
  await post('/insert', { metadata: 'Investment Banking: mergers, acquisitions and bond issuance', category: 'finance' });

  console.log('3. Searching for "stock market dividend"...');
  const searchRes = await post('/search', {
    query: 'stock market dividend',
    algo: 'ivf',
    metric: 'cosine',
    k: 3,
    alpha: -1.0
  });
  console.log(`Auto Alpha: ${searchRes.effectiveAlpha} (${searchRes.alphaMode})`);
  searchRes.results.forEach(r => console.log(`  [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)})`));
}

testCreateCluster();
