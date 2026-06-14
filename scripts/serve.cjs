// Static file server with Data/Image/ support
const http = require('http');
const fs = require('fs');
const path = require('path');
const publicRoot = path.join(__dirname, '..', 'public');
const dataImageRoot = path.join(__dirname, '..', 'Data', 'Image');
const port = parseInt(process.argv[2] || '8080');

const mime = {
  '.html': 'text/html', '.js': 'application/javascript',
  '.json': 'application/json', '.bin': 'application/octet-stream',
  '.txt': 'text/plain', '.map': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.tif': 'image/tiff', '.tiff': 'image/tiff',
  '.svg': 'image/svg+xml', '.css': 'text/css', '.wasm': 'application/wasm',
};

http.createServer((req, res) => {
  let url = req.url.split('?')[0];

  // API: list images
  if (url === '/api/images') {
    fs.readdir(dataImageRoot, (err, files) => {
      if (err) { res.writeHead(500); res.end('[]'); return; }
      const images = files
        .filter(f => /\.(png|jpg|jpeg|tif|tiff)$/i.test(f))
        .map(f => ({ name: f, url: '/data-image/' + encodeURIComponent(f) }));
      res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      res.end(JSON.stringify(images));
    });
    return;
  }

  if (url.startsWith('/data-image/')) {
    const f = decodeURIComponent(url.slice('/data-image/'.length));
    const fp = path.join(dataImageRoot, f);
    // Prevent directory traversal
    if (fp.indexOf(dataImageRoot) !== 0) { res.writeHead(403); res.end('Forbidden'); return; }
    fs.readFile(fp, (err, data) => {
      if (err) { res.writeHead(404); res.end('Not found'); return; }
      const ext = path.extname(fp).toLowerCase();
      res.writeHead(200, { 'Content-Type': mime[ext] || 'application/octet-stream', 'Access-Control-Allow-Origin': '*' });
      res.end(data);
    });
    return;
  }

  // Default: serve from public/
  if (url.endsWith('/')) url += 'index.html';
  const fp = path.join(publicRoot, url);
  // Prevent directory traversal
  if (fp.indexOf(publicRoot) !== 0) { res.writeHead(403); res.end('Forbidden'); return; }
  fs.readFile(fp, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found: ' + fp); return; }
    const ext = path.extname(fp).toLowerCase();
    res.writeHead(200, {
      'Content-Type': mime[ext] || 'application/octet-stream',
      'Access-Control-Allow-Origin': '*',
    });
    res.end(data);
  });
}).listen(port, () => console.log(`Serving on http://localhost:${port}`));
