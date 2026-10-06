const http = require('http');

function search(query, alpha = -1.0) {
  const data = JSON.stringify({
    query: query,
    algo: "hnsw",
    metric: "cosine",
    k: 4,
    alpha: alpha
  });

  const req = http.request({
    hostname: 'localhost',
    port: 8080,
    path: '/search',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': data.length
    }
  }, (res) => {
    let raw = '';
    res.on('data', chunk => raw += chunk);
    res.on('end', () => {
      console.log(`\n=== QUERY: "${query}" (Alpha: ${alpha}) ===`);
      const parsed = JSON.parse(raw);
      console.log(`Effective Alpha: ${parsed.effectiveAlpha} (${parsed.alphaMode}) | Latency: ${parsed.latencyUs}us`);
      parsed.results.forEach((r, i) => {
        console.log(`  #${i+1} [${r.category}] ${r.metadata} (Dist: ${r.distance.toFixed(4)}, Hybrid: ${(r.hybridScore*100).toFixed(1)}%)`);
      });
    });
  });

  req.write(data);
  req.end();
}

search("delicious pizza and spicy taco", -1.0);
setTimeout(() => search("asteroid and solar planet orbit", -1.0), 300);
setTimeout(() => search("basketball tournament and wimbledon match", -1.0), 600);
setTimeout(() => search("binary search tree and dynamic programming", -1.0), 900);
setTimeout(() => search("calculus derivative and matrix eigenvalues", -1.0), 1200);
