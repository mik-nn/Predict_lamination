import * as esbuild from 'esbuild';

await esbuild.build({
  entryPoints: ['src/app.ts'],
  outfile: 'public/dist/app.js',
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2020',
  sourcemap: true,
  minify: true,
  external: ['@tensorflow/tfjs'],
});

console.log('→ public/dist/app.js');
