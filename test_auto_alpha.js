const http = require('http');

function search(label, query, alpha = -1.0) {
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
      console.log(`\n=== [${label}] QUERY: "${query}" ===`);
      const parsed = JSON.parse(raw);
      console.log(`Auto Alpha: ${parsed.effectiveAlpha} (${parsed.alphaMode})`);
      parsed.results.forEach((r, i) => {
        console.log(`  #${i+1} [${r.category}] ${r.metadata} (d=${r.distance.toFixed(4)}, sparse=${r.sparseScore.toFixed(3)}, hybrid=${(r.hybridScore*100).toFixed(1)}%)`);
      });
    });
  });

  req.write(data);
  req.end();
}

// 1. Exact BM25 keyword heavy query
search("EXACT KEYWORD QUERY", "Neapolitan Pizza wood-fired dough", -1.0);

// 2. Pure semantic / conceptual query without exact keyword matches
setTimeout(() => search("CONCEPTUAL SEMANTIC QUERY", "gourmet Italian oven baked dish with melted mozzarella", -1.0), 300);

// 3. Space exploration conceptual query
setTimeout(() => search("SPACE CONCEPTUAL", "interplanetary orbit telescope looking at cosmos", -1.0), 600);
