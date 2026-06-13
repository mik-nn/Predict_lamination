// Re-export for Node.js scripts (imported via tsx which resolves .js→.ts)
// This file exists so that existing scripts importing "../src/cgats-parser.js"
// get the Node.js version (with parseCgatsFile).
export { parseCgatsText, parseCgatsFile } from './cgats-parser.node.ts';
export { parseCGATS, extractSpectralData } from './strip-matcher.ts';
