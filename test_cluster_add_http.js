const http = require('http');

async function test() {
  const data = JSON.stringify({
    name: 'space_science',
    label: 'Space Science',
    color: '#38bdf8'
  });

  const req = http.request({
    hostname: '127.0.0.1',
    port: 8080,
    path: '/cluster/add',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(data)
    }
  }, res => {
    let b = '';
    res.on('data', d => b += d);
    res.on('end', () => console.log('Cluster Add Res:', res.statusCode, b));
  });

  req.on('error', e => console.error('Req error:', e.message));
  req.write(data);
  req.end();
}

test();
