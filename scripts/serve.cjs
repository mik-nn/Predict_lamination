// Simple static file server for testing
const http = require('http');
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', 'public');
const port = parseInt(process.argv[2] || '8080');

const mime = {
  '.html': 'text/html', '.js': 'application/javascript',
  '.json': 'application/json', '.bin': 'application/octet-stream',
  '.txt': 'text/plain', '.map': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.css': 'text/css', '.wasm': 'application/wasm',
};

http.createServer((req, res) => {
  let f = req.url.split('?')[0];
  if (f.endsWith('/')) f += 'index.html';
  const fp = path.join(root, f);
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found: ' + fp); return; }
    const ext = path.extname(fp).toLowerCase();
    res.writeHead(200, {
      'Content-Type': mime[ext] || 'application/octet-stream',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(data);
  });
}).listen(port, () => console.log(`Serving public/ on http://localhost:${port}`));
