const fs = require('fs');

function applyCurve(curve, value) {
  const idx = value / 65535 * (curve.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, curve.length - 1);
  const frac = idx - lo;
  return curve[lo] * (1 - frac) + curve[hi] * frac;
}

function main() {
  const iccPath = process.argv[2] || 'scripts/ISOcoated_v2_300_eci.icc';
  const outputPath = process.argv[3] || 'public/cmyk-lab-lut.json';

  const buf = fs.readFileSync(iccPath);
  const tagCount = buf.readUInt32BE(128);
  let a2b0_off, a2b0_sz;
  for (let i = 0; i < tagCount; i++) {
    const sig = buf.slice(132 + i * 12, 132 + i * 12 + 4).toString('ascii');
    if (sig === 'A2B0') {
      a2b0_off = buf.readUInt32BE(132 + i * 12 + 4);
      a2b0_sz = buf.readUInt32BE(132 + i * 12 + 8);
      break;
    }
  }
  if (!a2b0_off) throw new Error('A2B0 tag not found');

  const tag = buf.slice(a2b0_off, a2b0_off + a2b0_sz);
  const inputCh = tag[8];
  const outputCh = tag[9];
  const gridPts = tag[10];
  const TAG_HDR = 12;       // mft2 header: sig(4) + reserved(4) + inCh(1) + outCh(1) + grid(1) + reserved(1)
  const MATRIX_SIZE = 40;   // 9 × s15Fixed16 (36B) + encoding info (4B)
  const CURVE_ENTRIES = 256;
  const inCurveBytes = inputCh * CURVE_ENTRIES * 2;   // 4 × 512 = 2048
  const clutByteSize = Math.pow(gridPts, inputCh) * outputCh * 2; // 16^4 * 3 * 2 = 393216
  const outCurveBytes = outputCh * CURVE_ENTRIES * 2; // 3 × 512 = 1536

  let offset = TAG_HDR + MATRIX_SIZE; // 52

  // Read input curves (256 entries per channel)
  const inCurves = [];
  for (let ch = 0; ch < inputCh; ch++) {
    const curve = [];
    for (let i = 0; i < CURVE_ENTRIES; i++) {
      curve.push(tag.readUInt16BE(offset + i * 2));
    }
    offset += CURVE_ENTRIES * 2;
    inCurves.push(curve);
  }

  // Verify CLUT start
  const clutStart = offset;
  console.log(`CLUT starts at offset ${clutStart} (expected 2100)`);

  // Read CLUT
  const clutGridSize = Math.pow(gridPts, inputCh);
  const clutData = tag.slice(offset, offset + clutByteSize);
  offset += clutByteSize;

  // Read output curves (256 entries per channel)
  const outCurves = [];
  for (let ch = 0; ch < outputCh; ch++) {
    const curve = [];
    for (let i = 0; i < CURVE_ENTRIES; i++) {
      curve.push(tag.readUInt16BE(offset + i * 2));
    }
    offset += CURVE_ENTRIES * 2;
    outCurves.push(curve);
  }

  console.log(`Grid: ${gridPts}^${inputCh}=${clutGridSize}, inCurves ok, outCurves ok, total consumed=${offset}`);

  // Pre-compute Lab values for each grid point
  const values = [];
  for (let gi = 0; gi < clutGridSize; gi++) {
    const raw0 = clutData.readUInt16BE(gi * 6);
    const raw1 = clutData.readUInt16BE(gi * 6 + 2);
    const raw2 = clutData.readUInt16BE(gi * 6 + 4);

    const m0 = applyCurve(outCurves[0], raw0);
    const m1 = applyCurve(outCurves[1], raw1);
    const m2 = applyCurve(outCurves[2], raw2);

    const L = m0 / 65535 * 100;
    const a = m1 / 65535 * 255 - 128;
    const b = m2 / 65535 * 255 - 128;

    values.push(L, a, b);
  }

  // Reorder input curves from ICC order [K, Y, M, C] to standard [C, M, Y, K]
  const orderedCurves = [inCurves[3], inCurves[2], inCurves[1], inCurves[0]];

  // Reorder values from ICC grid (C*g^3 + M*g^2 + Y*g + K) to standard CMYK (K*g^3 + Y*g^2 + M*g + C)
  const g = gridPts;
  const reordered = new Array(clutGridSize * 3);
  for (let C = 0; C < g; C++)
    for (let M = 0; M < g; M++)
      for (let Y = 0; Y < g; Y++)
        for (let K = 0; K < g; K++) {
          const iccIdx = (C * g * g * g + M * g * g + Y * g + K) * 3;
          const cmykIdx = (K * g * g * g + Y * g * g + M * g + C) * 3;
          reordered[cmykIdx] = values[iccIdx];
          reordered[cmykIdx + 1] = values[iccIdx + 1];
          reordered[cmykIdx + 2] = values[iccIdx + 2];
        }

  const lut = { gridPts, inputCurves: orderedCurves, values: Array.from(reordered) };
  const json = JSON.stringify(lut);
  fs.writeFileSync(outputPath, json);
  console.log(`Written ${outputPath} (${Math.round(json.length / 1024)} KB, ${reordered.length / 3} grid pts)`);

  // Verify in standard CMYK order: index = ((K*g + Y)*g + M)*g + C
  const v = (cmyk) => {
    const [C, M, Y, K] = cmyk;
    const idx = (((K * g + Y) * g + M) * g + C) * 3;
    return [reordered[idx], reordered[idx + 1], reordered[idx + 2]];
  };

  console.log('CMYK[0,0,0,0] → Lab:', v([0,0,0,0]).map(x => x.toFixed(2)));
  console.log('CMYK[15,0,0,0] → Lab:', v([15,0,0,0]).map(x => x.toFixed(2)));
  console.log('CMYK[0,15,0,0] → Lab:', v([0,15,0,0]).map(x => x.toFixed(2)));
  console.log('CMYK[0,0,15,0] → Lab:', v([0,0,15,0]).map(x => x.toFixed(2)));
  console.log('CMYK[0,0,0,15] → Lab:', v([0,0,0,15]).map(x => x.toFixed(2)));
  console.log('CMYK[15,15,15,15] → Lab:', v([15,15,15,15]).map(x => x.toFixed(2)));
}

main();
