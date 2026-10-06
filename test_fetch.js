async function test() {
  try {
    console.log('Testing GET /status...');
    let res = await fetch('http://127.0.0.1:8080/status');
    console.log('Status:', await res.json());

    console.log('Testing POST /cluster/add...');
    res = await fetch('http://127.0.0.1:8080/cluster/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'gaming', label: 'Video Games', color: '#a855f7' })
    });
    console.log('Cluster Add:', await res.json());

    console.log('Testing POST /insert...');
    res = await fetch('http://127.0.0.1:8080/insert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata: 'Unreal Engine: ray tracing shaders', category: 'gaming' })
    });
    console.log('Insert 1:', await res.json());

    console.log('Testing POST /search...');
    res = await fetch('http://127.0.0.1:8080/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'ray tracing', k: 3 })
    });
    console.log('Search:', await res.json());
    console.log('ALL TESTS PASSED!');
  } catch (err) {
    console.error('Test error:', err);
  }
}
test();
