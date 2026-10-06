const http = require('http');

function postSearch() {
  const data = JSON.stringify({
    query: "binary tree",
    algo: "hnsw",
    metric: "cosine",
    k: 3,
    alpha: -1.0
  });

  const req = http.request({
    hostname: '127.0.0.1',
    port: 8080,
    path: '/search',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(data)
    }
  }, (res) => {
    let body = '';
    res.on('data', chunk => body += chunk);
    res.on('end', () => {
      console.log('STATUS:', res.statusCode);
      console.log('RESPONSE:', body);
    });
  });

  req.on('error', (e) => {
    console.error('REQUEST ERROR:', e.message);
  });

  req.write(data);
  req.end();
}

postSearch();
