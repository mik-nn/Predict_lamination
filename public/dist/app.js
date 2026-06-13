// src/icc-writer.ts
function u16(v) {
  return [v >> 8 & 255, v & 255];
}
function u32(v) {
  return [v >> 24 & 255, v >> 16 & 255, v >> 8 & 255, v & 255];
}
function u8ArrayFromStrings(...strs) {
  const out = [];
  for (const s of strs) for (let i = 0; i < s.length; i++) out.push(s.charCodeAt(i));
  return out;
}
function labToLab8(L, a, b) {
  return [
    Math.round(Math.max(0, Math.min(100, L)) / 100 * 255),
    Math.round(Math.max(-128, Math.min(127, a)) + 128),
    Math.round(Math.max(-128, Math.min(127, b)) + 128)
  ];
}
function identityLUT() {
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) lut[i] = i;
  return lut;
}
function u1Fixed15(v) {
  return u16(Math.round(v * 32768));
}
function buildLUT8Tag(params) {
  const ic = params.inputChannels;
  const oc = params.outputChannels;
  const gp = params.clutPoints;
  const clut = params.clutData;
  const matrixLen = 9 * 2;
  const inputLUTLen = ic * 256;
  const clutLen = Math.pow(gp, ic) * oc;
  const outputLUTLen = oc * 256;
  const tagLen = 4 + 4 + 1 + 1 + 1 + 1 + matrixLen + inputLUTLen + clutLen + outputLUTLen;
  if (clut.length !== clutLen) {
    throw new Error(`CLUT data length ${clut.length} !== expected ${clutLen} (${gp}^${ic} \xD7 ${oc})`);
  }
  const buf = new Uint8Array(tagLen);
  let off = 0;
  buf.set([109, 102, 116, 49], off);
  off += 4;
  buf.set([0, 0, 0, 0], off);
  off += 4;
  buf[off++] = ic;
  buf[off++] = oc;
  buf[off++] = gp;
  buf[off++] = 0;
  if (ic >= 1 && ic <= 15) {
    const ident3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (const v of ident3) {
      buf.set(u1Fixed15(v), off);
      off += 2;
    }
  } else {
    for (let i = 0; i < 9; i++) {
      buf.set(u16(0), off);
      off += 2;
    }
  }
  for (let c = 0; c < ic; c++) {
    const lut = identityLUT();
    buf.set(lut, off);
    off += 256;
  }
  buf.set(clut, off);
  off += clutLen;
  for (let c = 0; c < oc; c++) {
    const lut = identityLUT();
    buf.set(lut, off);
    off += 256;
  }
  return buf;
}
function buildDeviceLink(params) {
  const desc = params.description || "Unlaminated\u2192Laminated DeviceLink";
  const ic = params.inputChannels;
  const oc = params.outputChannels;
  const a2b0Tag = buildLUT8Tag(params);
  const tagCount = 1;
  const tagTableStart = 128;
  const tagTableLen = 4 + tagCount * 12;
  const a2b0Offset = tagTableStart + tagTableLen;
  const profileSize = a2b0Offset + a2b0Tag.length;
  const header = new Uint8Array(128);
  let off = 0;
  header.set(u32(profileSize), off);
  off += 4;
  header.set(u8ArrayFromStrings("none"), off);
  off += 4;
  header.set([2, 64, 0, 0], off);
  off += 4;
  header.set(u8ArrayFromStrings("link"), off);
  off += 4;
  header.set(u8ArrayFromStrings(ic === 4 ? "CMYK" : "Lab "), off);
  off += 4;
  header.set(u8ArrayFromStrings(oc === 4 ? "CMYK" : "Lab "), off);
  off += 4;
  header.set(u16(2024), off);
  off += 2;
  header.set(u16(1), off);
  off += 2;
  header.set(u16(1), off);
  off += 2;
  header.set(u16(0), off);
  off += 2;
  header.set(u16(0), off);
  off += 2;
  header.set(u16(0), off);
  off += 2;
  header.set(u8ArrayFromStrings("acsp"), off);
  off += 4;
  header.set(u8ArrayFromStrings("APPL"), off);
  off += 4;
  header.set(u32(0), off);
  off += 4;
  header.set(u32(0), off);
  off += 4;
  header.set(u32(0), off);
  off += 4;
  header.set(u32(0), off);
  off += 4;
  header.set(u32(0), off);
  off += 4;
  header.set(u32(0), off);
  off += 4;
  const d50X = Math.round(0.9642 * 65536);
  const d50Y = Math.round(1 * 65536);
  const d50Z = Math.round(0.8249 * 65536);
  header.set(u32(d50X), off);
  off += 4;
  header.set(u32(d50Y), off);
  off += 4;
  header.set(u32(d50Z), off);
  off += 4;
  header.set(u8ArrayFromStrings("none"), off);
  off += 4;
  for (let i = 0; i < 44; i++) header[off++] = 0;
  const tagTable = new Uint8Array(tagTableLen);
  let toff = 0;
  tagTable.set(u32(tagCount), toff);
  toff += 4;
  tagTable.set(u8ArrayFromStrings("A2B0"), toff);
  toff += 4;
  tagTable.set(u32(a2b0Offset), toff);
  toff += 4;
  tagTable.set(u32(a2b0Tag.length), toff);
  toff += 4;
  const profile = new Uint8Array(profileSize);
  profile.set(header, 0);
  profile.set(tagTable, tagTableStart);
  profile.set(a2b0Tag, a2b0Offset);
  return profile.buffer;
}
function allocateCLUT(inputChannels, outputChannels, clutPoints) {
  const size = Math.pow(clutPoints, inputChannels) * outputChannels;
  const array = new Uint8Array(size);
  function linearIndex(indices) {
    let idx = 0;
    for (let c = 0; c < inputChannels; c++) {
      idx = idx * clutPoints + indices[c];
    }
    return idx * outputChannels;
  }
  return {
    array,
    setter: (indices, values) => {
      const base = linearIndex(indices);
      for (let k = 0; k < outputChannels; k++) array[base + k] = values[k];
    },
    getter: (indices) => {
      const base = linearIndex(indices);
      const out = [];
      for (let k = 0; k < outputChannels; k++) out.push(array[base + k]);
      return out;
    }
  };
}

// src/strip-matcher.ts
function parseCGATS(text) {
  const lines = text.split(/\r?\n/);
  let header = {};
  let columns = [];
  let data = [];
  let stage = "header";
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    if (line.startsWith("BEGIN_DATA_FORMAT")) {
      stage = "format";
      columns = [];
      continue;
    }
    if (line.startsWith("END_DATA_FORMAT")) {
      stage = "header";
      continue;
    }
    if (line.startsWith("BEGIN_DATA")) {
      stage = "data";
      continue;
    }
    if (line.startsWith("END_DATA")) break;
    if (stage === "format" && columns.length === 0) {
      columns = line.split(/\s+/);
      continue;
    }
    if (stage === "data" && columns.length > 0) {
      const values = line.split(/\s+/);
      if (values.length >= columns.length) {
        const entry = {};
        for (let j = 0; j < columns.length; j++) {
          entry[columns[j]] = parseFloat(values[j]);
        }
        data.push(entry);
      }
      continue;
    }
    if (stage === "header" && line.includes("	") === false) {
      const m = line.match(/^(\w+)\s+"(.*)"$/);
      if (m) header[m[1]] = m[2];
      else {
        const m2 = line.match(/^(\w+)\s+(\S+)/);
        if (m2) header[m2[1]] = m2[2];
      }
    }
  }
  return { header, data, columns };
}
function computeRows(totalPatches, totalRows) {
  const patchesPerRow = Math.ceil(totalPatches / totalRows);
  const rows = [];
  let start = 0;
  for (let r = 0; r < totalRows; r++) {
    const end = Math.min(start + patchesPerRow, totalPatches);
    if (start >= totalPatches) break;
    rows.push({ start, end, count: end - start });
    start = end;
  }
  return { rows, patchesPerRow };
}
function getPatchIndicesInRow(row, patchesPerRow, totalPatches) {
  const start = row * patchesPerRow;
  const end = Math.min(start + patchesPerRow, totalPatches);
  const indices = [];
  for (let i = start; i < end; i++) indices.push(i);
  return indices;
}
function extractSpectralData(data, columns) {
  const spectralCols = columns.filter((c) => /^SPECTRAL_NM_\d+$/.test(c) || /^S\d{3}$/.test(c) || c.startsWith("SPEC_"));
  const spectral = [];
  const cmyk = [];
  const lab = [];
  for (const entry of data) {
    const s = spectralCols.map((c) => entry[c]);
    if (s.length > 0) spectral.push(s);
    const cyan = entry["CMYK_C"] ?? entry["C"] ?? entry["CYAN"] ?? NaN;
    const magenta = entry["CMYK_M"] ?? entry["M"] ?? entry["MAGENTA"] ?? NaN;
    const yellow = entry["CMYK_Y"] ?? entry["Y"] ?? entry["YELLOW"] ?? NaN;
    const black = entry["CMYK_K"] ?? entry["K"] ?? entry["BLACK"] ?? NaN;
    if (!isNaN(cyan)) cmyk.push([cyan, magenta, yellow, black]);
    const l = entry["LAB_L"] ?? entry["L"] ?? entry["L*"] ?? NaN;
    const a = entry["LAB_A"] ?? entry["a"] ?? entry["a*"] ?? NaN;
    const b = entry["LAB_B"] ?? entry["b"] ?? entry["b*"] ?? NaN;
    if (!isNaN(l)) lab.push([l, a, b]);
  }
  return { spectral, cmyk, lab };
}
function featureDiversityScore(features) {
  const n = features.length;
  if (n < 2) return 0;
  let totalDist = 0;
  let pairs = 0;
  const step = Math.max(1, Math.floor(n / 100));
  for (let i = 0; i < n; i += step) {
    for (let j = i + step; j < n; j += step) {
      let d2 = 0;
      for (let k = 0; k < features[i].length; k++) {
        const d = features[i][k] - features[j][k];
        d2 += d * d;
      }
      totalDist += Math.sqrt(d2);
      pairs++;
    }
  }
  return totalDist / pairs;
}
function rowDiversityScores(features, patchesPerRow, totalPatches, topN) {
  const totalRows = Math.ceil(totalPatches / patchesPerRow);
  const results = [];
  for (let r = 0; r < totalRows; r++) {
    const indices = getPatchIndicesInRow(r, patchesPerRow, totalPatches);
    if (indices.length === 0) continue;
    const rowFeatures = [];
    for (const idx of indices) {
      if (idx < features.length) rowFeatures.push(features[idx]);
    }
    if (rowFeatures.length < 2) {
      results.push({ rowIndex: r, patchCount: rowFeatures.length, diversity: 0, isRecommended: false });
      continue;
    }
    const div = featureDiversityScore(rowFeatures);
    results.push({ rowIndex: r, patchCount: rowFeatures.length, diversity: div, isRecommended: false });
  }
  const sorted = [...results].sort((a, b) => b.diversity - a.diversity);
  const nMark = topN ?? 1;
  for (let i = 0; i < Math.min(nMark, sorted.length); i++) {
    const r = results.find((x) => x.rowIndex === sorted[i].rowIndex);
    if (r) r.isRecommended = true;
  }
  return results;
}
function generateSubsetCGATS(originalText, rowIndices, patchesPerRow, totalPatches) {
  const lines = originalText.split(/\r?\n/);
  let dataStart = -1;
  let dataEnd = -1;
  let numberSetsIdx = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "BEGIN_DATA") dataStart = i;
    if (t === "END_DATA" && dataStart !== -1) {
      dataEnd = i;
      break;
    }
    if (/^NUMBER_OF_SETS\b/.test(t)) numberSetsIdx = i;
  }
  if (dataStart === -1 || dataEnd === -1) {
    throw new Error("Could not find BEGIN_DATA / END_DATA in CGATS file");
  }
  const selectedIndices = /* @__PURE__ */ new Set();
  for (const r of rowIndices) {
    const indices = getPatchIndicesInRow(r, patchesPerRow, totalPatches);
    for (const idx of indices) selectedIndices.add(idx);
  }
  const out = [];
  for (let i = 0; i < dataStart; i++) {
    if (i === numberSetsIdx) {
      out.push(`NUMBER_OF_SETS	${selectedIndices.size}`);
    } else {
      out.push(lines[i]);
    }
  }
  out.push("BEGIN_DATA");
  let dataRow = 0;
  for (let i = dataStart + 1; i < dataEnd; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    if (selectedIndices.has(dataRow)) {
      out.push(lines[i]);
    }
    dataRow++;
  }
  out.push("END_DATA");
  for (let i = dataEnd + 1; i < lines.length; i++) {
    out.push(lines[i]);
  }
  return out.join("\n");
}
function verifySubsetMatch(uPatches, lPatches, expectedRowIndices, patchesPerRow) {
  const expectedCMYKeys = /* @__PURE__ */ new Map();
  for (const r of expectedRowIndices) {
    const indices = getPatchIndicesInRow(r, patchesPerRow, uPatches.length);
    for (const idx of indices) {
      const k = uPatches[idx].cmyk.join(",");
      expectedCMYKeys.set(k, (expectedCMYKeys.get(k) ?? 0) + 1);
    }
  }
  const remaining = new Map(expectedCMYKeys);
  const missing = [];
  const extra = [];
  let matched = 0;
  for (const p of lPatches) {
    const k = p.cmyk.join(",");
    const remainingCount = remaining.get(k) ?? 0;
    if (remainingCount > 0) {
      remaining.set(k, remainingCount - 1);
      matched++;
    } else {
      extra.push({ sampleId: p.sampleId, cmyk: Array.from(p.cmyk) });
    }
  }
  for (const [k, count] of remaining) {
    if (count > 0) {
      missing.push({ sampleId: "", cmyk: k.split(",").map(Number) });
    }
  }
  const expected = lPatches.length;
  const ok = extra.length === 0 && matched === lPatches.length;
  return { matched, expected, missing, extra, ok };
}

// src/cgats-parser.ts
var SPECTRAL_START = 380;
var SPECTRAL_STEP = 10;
function parseCgatsText(text) {
  const lines = text.split(/\r?\n/);
  let dataFormatLine = -1;
  let dataStartLine = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === "BEGIN_DATA_FORMAT") dataFormatLine = i;
    if (t === "BEGIN_DATA") dataStartLine = i;
  }
  if (dataFormatLine === -1 || dataStartLine === -1) {
    throw new Error("Failed to find BEGIN_DATA_FORMAT or BEGIN_DATA");
  }
  const headerLine = lines[dataFormatLine + 1].trim();
  const fieldNames = headerLine.split(/\t/);
  const sampleIdIdx = fieldNames.indexOf("SAMPLE_ID");
  const cIdx = fieldNames.indexOf("CMYK_C");
  const mIdx = fieldNames.indexOf("CMYK_M");
  const yIdx = fieldNames.indexOf("CMYK_Y");
  const kIdx = fieldNames.indexOf("CMYK_K");
  const spectralIdxs = [];
  for (let w = 0; w < 36; w++) {
    const nm = SPECTRAL_START + w * SPECTRAL_STEP;
    const col = `SPECTRAL_NM_${nm}`;
    const idx = fieldNames.indexOf(col);
    if (idx !== -1) spectralIdxs.push(idx);
  }
  const patches = [];
  for (let i = dataStartLine + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "END_DATA" || line.length === 0) break;
    const parts = line.split(/\t/);
    if (parts.length < fieldNames.length) continue;
    const sampleId = sampleIdIdx !== -1 ? parts[sampleIdIdx] : `${i}`;
    const cmyk = [
      cIdx !== -1 ? parseFloat(parts[cIdx]) : 0,
      mIdx !== -1 ? parseFloat(parts[mIdx]) : 0,
      yIdx !== -1 ? parseFloat(parts[yIdx]) : 0,
      kIdx !== -1 ? parseFloat(parts[kIdx]) : 0
    ];
    const spectra = new Float64Array(36);
    for (let w = 0; w < spectralIdxs.length; w++) {
      spectra[w] = parseFloat(parts[spectralIdxs[w]]);
    }
    patches.push({ sampleId, cmyk, spectra });
  }
  return patches;
}

// src/color-math.ts
var D50_XYZ = [0.9642, 1, 0.8249];
var deg2rad = Math.PI / 180;
function xyzToLab(x, y, z) {
  const f = (t) => t > 8856e-6 ? Math.cbrt(t) : (903.3 * t + 16) / 116;
  const fx = f(x / D50_XYZ[0]);
  const fy = f(y);
  const fz = f(z / D50_XYZ[2]);
  const L = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const b = 200 * (fy - fz);
  return [L, a, b];
}
var D50_SPECTRAL = [
  49.9755,
  54.6482,
  82.7549,
  91.486,
  93.4318,
  86.6823,
  104.865,
  117.008,
  117.812,
  114.861,
  115.923,
  108.811,
  109.35,
  107.802,
  104.79,
  107.689,
  104.406,
  97.84,
  96.334,
  96.614,
  92.435,
  89.492,
  96.259,
  90.064,
  89.707,
  92.244,
  90.988,
  89.277,
  91.405,
  91.015,
  89.195,
  87.515,
  92.155,
  87.983,
  84.881,
  82.288
];
var CIE_CMF_X = [
  169e-6,
  2361e-6,
  0.01911,
  0.08474,
  0.2045,
  0.3147,
  0.3837,
  0.3707,
  0.3023,
  0.1956,
  0.08051,
  0.01617,
  3817e-6,
  0.01567,
  0.03747,
  0.04865,
  0.04973,
  0.04583,
  0.03801,
  0.02867,
  0.01984,
  0.01264,
  7417e-6,
  4032e-6,
  2184e-6,
  1193e-6,
  589e-6,
  325e-6,
  176e-6,
  97e-6,
  53e-6,
  28e-6,
  15e-6,
  9e-6,
  4e-6,
  2e-6
];
var CIE_CMF_Y = [
  2e-6,
  32e-6,
  265e-6,
  1503e-6,
  6809e-6,
  0.01872,
  0.04844,
  0.08984,
  0.1282,
  0.1275,
  0.09146,
  0.04351,
  0.01464,
  3575e-6,
  1294e-6,
  3468e-6,
  7966e-6,
  0.01086,
  0.01115,
  9887e-6,
  7827e-6,
  5646e-6,
  3694e-6,
  2148e-6,
  1196e-6,
  662e-6,
  37e-5,
  2e-4,
  108e-6,
  59e-6,
  32e-6,
  17e-6,
  1e-5,
  5e-6,
  3e-6,
  1e-6
];
var CIE_CMF_Z = [
  75e-5,
  0.01069,
  0.08601,
  0.3895,
  0.9725,
  1.5535,
  1.9673,
  1.9948,
  1.7454,
  1.3176,
  0.77213,
  0.4149,
  0.3837,
  0.3864,
  0.3866,
  0.3792,
  0.3631,
  0.329,
  0.2795,
  0.2158,
  0.1571,
  0.108,
  0.07098,
  0.04368,
  0.02535,
  0.01433,
  8049e-6,
  452e-5,
  2521e-6,
  1401e-6,
  776e-6,
  434e-6,
  249e-6,
  137e-6,
  75e-6,
  4e-5
];
var NORM_D50 = D50_SPECTRAL.reduce((s, d, i) => s + d * CIE_CMF_Y[i], 0);
function spectralToXYZ(reflectance) {
  if (reflectance.length !== 36) {
    throw new Error(`Expected 36-channel spectral data, got ${reflectance.length}`);
  }
  let X = 0, Y = 0, Z = 0;
  for (let i = 0; i < 36; i++) {
    const r = reflectance[i];
    X += r * D50_SPECTRAL[i] * CIE_CMF_X[i];
    Y += r * D50_SPECTRAL[i] * CIE_CMF_Y[i];
    Z += r * D50_SPECTRAL[i] * CIE_CMF_Z[i];
  }
  X /= NORM_D50;
  Y /= NORM_D50;
  Z /= NORM_D50;
  return [X, Y, Z];
}
function deltaE00(L1, a1, b1, L2, a2, b2) {
  const L_avg = (L1 + L2) / 2;
  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const C_avg = (C1 + C2) / 2;
  const G = 0.5 * (1 - Math.sqrt(Math.pow(C_avg, 7) / (Math.pow(C_avg, 7) + Math.pow(25, 7))));
  const a1p = a1 * (1 + G);
  const a2p = a2 * (1 + G);
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);
  const Cp_avg = (C1p + C2p) / 2;
  const h1p = Math.atan2(b1, a1p) * (180 / Math.PI);
  const h2p = Math.atan2(b2, a2p) * (180 / Math.PI);
  const h1pd = h1p < 0 ? h1p + 360 : h1p;
  const h2pd = h2p < 0 ? h2p + 360 : h2p;
  const dh = Math.abs(h1pd - h2pd) > 180 ? h2pd - h1pd + 360 * (h2pd <= h1pd ? 1 : -1) : h2pd - h1pd;
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(dh * deg2rad / 2);
  const dL = L2 - L1;
  const dCp = C2p - C1p;
  const Lp_avg = (L1 + L2) / 2;
  const Cp_avg2 = (C1p + C2p) / 2;
  const hp_avg = Math.abs(h1pd - h2pd) > 180 ? (h1pd + h2pd + 360) / 2 : (h1pd + h2pd) / 2;
  const T = 1 - 0.17 * Math.cos((hp_avg - 30) * deg2rad) + 0.24 * Math.cos(2 * hp_avg * deg2rad) + 0.32 * Math.cos((3 * hp_avg + 6) * deg2rad) - 0.2 * Math.cos((4 * hp_avg - 63) * deg2rad);
  const SL = 1 + 0.015 * Math.pow(Lp_avg - 50, 2) / Math.sqrt(20 + Math.pow(Lp_avg - 50, 2));
  const SC = 1 + 0.045 * Cp_avg2;
  const SH = 1 + 0.015 * Cp_avg2 * T;
  const RC = 2 * Math.sqrt(Math.pow(Cp_avg2, 7) / (Math.pow(Cp_avg2, 7) + Math.pow(25, 7)));
  const dTheta = 30 * Math.exp(-Math.pow((hp_avg - 275) / 25, 2));
  const RT = -Math.sin(2 * dTheta * deg2rad) * RC;
  const de = Math.sqrt(
    Math.pow(dL / SL, 2) + Math.pow(dCp / SC, 2) + Math.pow(dHp / SH, 2) + RT * (dCp / SC) * (dHp / SH)
  );
  return de;
}

// src/ridge.ts
function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++)
        sum += L[i][k] * L[j][k];
      if (i === j)
        L[i][j] = Math.sqrt(A[i][i] - sum);
      else
        L[i][j] = (A[i][j] - sum) / L[j][j];
    }
  }
  return L;
}
function solveTriangular(L, b) {
  const n = L.length;
  const x = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++)
      sum += L[i][j] * x[j];
    x[i] = (b[i] - sum) / L[i][i];
  }
  return x;
}
function solveCholesky(A, b) {
  const L = cholesky(A);
  const n = L.length;
  const y = solveTriangular(L, b);
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++)
      sum += L[j][i] * x[j];
    x[i] = (y[i] - sum) / L[i][i];
  }
  return x;
}
function ridgeFit(X, Y, lambda = 0.01) {
  const N = X.length;
  const p = X[0].length;
  const q = Y[0].length;
  if (N === 0 || p === 0) throw new Error("Empty input");
  if (Y.length !== N) throw new Error("X and Y must have same number of rows");
  const XtX = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < p; j++) {
      const xij = X[i][j];
      if (xij === 0) continue;
      for (let k = 0; k < p; k++)
        XtX[j][k] += xij * X[i][k];
    }
  }
  for (let j = 0; j < p; j++)
    XtX[j][j] += lambda;
  const XtY = Array.from({ length: p }, () => new Array(q).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < p; j++) {
      const xij = X[i][j];
      if (xij === 0) continue;
      for (let k = 0; k < q; k++)
        XtY[j][k] += xij * Y[i][k];
    }
  }
  const W = Array.from({ length: p }, () => new Array(q));
  for (let k = 0; k < q; k++) {
    const b = XtY.map((row) => row[k]);
    const col = solveCholesky(XtX, b);
    for (let j = 0; j < p; j++)
      W[j][k] = col[j];
  }
  return { weights: W, lambda };
}
function ridgePredict(X, model) {
  const Y = [];
  for (const row of X) {
    const pred = new Array(model.weights[0].length).fill(0);
    for (let j = 0; j < model.weights.length; j++) {
      const xj = row[j];
      if (xj === 0) continue;
      for (let k = 0; k < model.weights[0].length; k++)
        pred[k] += xj * model.weights[j][k];
    }
    Y.push(pred);
  }
  return Y;
}

// src/app.ts
var _cache = null;
function clearCache() {
  _cache = null;
}
var BAKED = null;
async function loadBakedParams(url = "/baked-params.json") {
  const resp = await fetch(url);
  BAKED = await resp.json();
}
function getFeat(p) {
  const f = [];
  for (let w = 0; w < 36; w++) f.push(p.spectra[w]);
  for (let w = 0; w < 36; w++) f.push(p.spectra[w] * p.spectra[w]);
  return f;
}
function matchByCMYK(unlam, lam) {
  const lm = /* @__PURE__ */ new Map();
  for (const p of lam) {
    const k = p.cmyk.join(",");
    if (!lm.has(k)) lm.set(k, []);
    lm.get(k).push(p);
  }
  const out = [];
  for (const pu of unlam) {
    const k = pu.cmyk.join(",");
    const m = lm.get(k);
    if (m && m.length > 0) {
      out.push({ u: pu, l: m[0] });
      m.shift();
    }
  }
  return out;
}
function clamp(v) {
  return Math.max(0, Math.min(1, v));
}
function reconstructL(u, c, V) {
  const pred = new Float64Array(36);
  for (let w = 0; w < 36; w++) {
    let d = 0;
    for (let k = 0; k < 5; k++) d += c[k] * V[k][w];
    pred[w] = clamp(u[w] + d);
  }
  return pred;
}
function s2lab(s) {
  const [X, Y, Z] = spectralToXYZ(Array.from(s));
  return xyzToLab(X, Y, Z);
}
async function processDeviceLink(uText, lText, options = {}) {
  if (!BAKED) throw new Error("Baked params not loaded. Call loadBakedParams() first.");
  const { V, xm, xs, ym, ys } = BAKED;
  const gp = options.clutPoints ?? 17;
  const RANK = 5;
  const uPatches = parseCgatsText(uText);
  const lPatches = parseCgatsText(lText);
  const pairs = matchByCMYK(uPatches, lPatches);
  if (pairs.length < 10) {
    throw new Error(`Only ${pairs.length} matched pairs. Need at least 10.`);
  }
  const allFeatures = uPatches.map((p) => getFeat(p));
  const anchorFeatures = pairs.map((p) => getFeat(p.u));
  const normAll = allFeatures.map((r) => r.map((v, j) => (v - xm[j]) / xs[j]));
  const normAnchor = anchorFeatures.map((r) => r.map((v, j) => (v - xm[j]) / xs[j]));
  let model = options.model;
  if (!model) throw new Error("No model provided. Load TF.js model first.");
  const allT = globalThis.tf.tensor2d(normAll);
  const allPred = model.predict(allT);
  const allFrozenOut = (await allPred.array()).map((row) => row.map((v, k) => v * ys[k] + ym[k]));
  allT.dispose();
  allPred.dispose();
  const anchorT = globalThis.tf.tensor2d(normAnchor);
  const anchorPred = model.predict(anchorT);
  const anchorFrozenOut = (await anchorPred.array()).map((row) => row.map((v, k) => v * ys[k] + ym[k]));
  anchorT.dispose();
  anchorPred.dispose();
  const allDeltaC = pairs.map((p, i) => {
    const c = new Array(RANK).fill(0);
    for (let w = 0; w < 36; w++) {
      const d = p.l.spectra[w] - p.u.spectra[w];
      for (let k = 0; k < RANK; k++) c[k] += d * V[k][w];
    }
    return c;
  });
  const anchorDelta = allDeltaC.map(
    (c, i) => c.map((v, k) => v - anchorFrozenOut[i][k])
  );
  const ridgeModel = ridgeFit(anchorFrozenOut, anchorDelta, 0.01);
  const predictedLc = allFrozenOut.map((c, i) => {
    const predDelta = new Array(RANK).fill(0);
    for (let j = 0; j < RANK; j++) {
      const cj = c[j];
      if (cj === 0) continue;
      for (let k = 0; k < RANK; k++)
        predDelta[k] += cj * ridgeModel.weights[j][k];
    }
    return c.map((v, k) => v + predDelta[k]);
  });
  const predictedLab = predictedLc.map((c, i) => {
    const Lspec = reconstructL(uPatches[i].spectra, c, V);
    return s2lab(Lspec);
  });
  const actualLab = pairs.map((p) => s2lab(p.l.spectra));
  const anchorDe = [];
  const matchedUIndices = pairs.map((p) => uPatches.indexOf(p.u));
  for (let i = 0; i < pairs.length; i++) {
    const pi = matchedUIndices[i];
    if (pi === -1) continue;
    const predLab = predictedLab[pi];
    const actLab = actualLab[i];
    anchorDe.push(deltaE00(predLab[0], predLab[1], predLab[2], actLab[0], actLab[1], actLab[2]));
  }
  anchorDe.sort((a, b) => a - b);
  const uLab = uPatches.map((p) => s2lab(p.spectra));
  const Lmin = 0, Lmax = 100;
  const amin = -128, amax = 127;
  const bmin = -128, bmax = 127;
  const clut = allocateCLUT(3, 3, gp);
  for (let i = 0; i < gp; i++) {
    for (let j = 0; j < gp; j++) {
      for (let k = 0; k < gp; k++) {
        const Lt = Lmin + (Lmax - Lmin) * i / (gp - 1);
        const at = amin + (amax - amin) * j / (gp - 1);
        const bt = bmin + (bmax - bmin) * k / (gp - 1);
        let bestD = Infinity;
        let bestIdx = 0;
        for (let pi = 0; pi < uLab.length; pi++) {
          const d = Math.hypot(uLab[pi][0] - Lt, uLab[pi][1] - at, uLab[pi][2] - bt);
          if (d < bestD) {
            bestD = d;
            bestIdx = pi;
          }
        }
        const [L8, a8, b8] = labToLab8(
          predictedLab[bestIdx][0],
          predictedLab[bestIdx][1],
          predictedLab[bestIdx][2]
        );
        clut.setter([i, j, k], [L8, a8, b8]);
      }
    }
  }
  const stats = {
    totalPatches: uPatches.length,
    anchorPatches: pairs.length,
    medianDE: anchorDe[Math.floor(anchorDe.length / 2)] ?? 0,
    p95DE: anchorDe[Math.floor(anchorDe.length * 0.95)] ?? 0,
    maxDE: anchorDe[anchorDe.length - 1] ?? 0
  };
  const buffer = buildDeviceLink({
    inputChannels: 3,
    outputChannels: 3,
    clutPoints: gp,
    clutData: clut.array,
    description: `Unlaminated\u2192Laminated DeviceLink (${stats.anchorPatches} anchors, P95=${stats.p95DE.toFixed(2)})`
  });
  return { buffer, stats };
}
async function analyzeRows(uText, totalRows, model) {
  if (!BAKED) throw new Error("Baked params not loaded. Call loadBakedParams() first.");
  const { xm, xs, ym, ys, V } = BAKED;
  const RANK = 5;
  const uPatches = parseCgatsText(uText);
  const totalPatches = uPatches.length;
  const allFeatures = uPatches.map((p) => getFeat(p));
  const normAll = allFeatures.map((r) => r.map((v, j) => (v - xm[j]) / xs[j]));
  const t = globalThis.tf.tensor2d(normAll);
  const pred = model.predict(t);
  const arr = await pred.array();
  t.dispose();
  pred.dispose();
  const frozenCvals = arr.map((row) => row.map((v, k) => v * ys[k] + ym[k]));
  const { rows, patchesPerRow } = computeRows(totalPatches, totalRows);
  const diversity = rowDiversityScores(frozenCvals, patchesPerRow, totalPatches, 1);
  const allPatchesPerRow = Math.ceil(totalPatches / totalRows);
  const result = {
    totalPatches,
    patchesPerRow: allPatchesPerRow,
    rows: diversity.map((r) => ({
      index: r.rowIndex,
      patchCount: r.patchCount,
      diversity: r.diversity,
      isRecommended: r.isRecommended
    }))
  };
  return { result, uPatches, frozenCvals };
}
export {
  allocateCLUT,
  analyzeRows,
  buildDeviceLink,
  clearCache,
  computeRows,
  deltaE00,
  extractSpectralData,
  generateSubsetCGATS,
  labToLab8,
  loadBakedParams,
  parseCGATS,
  parseCgatsText,
  processDeviceLink,
  ridgeFit,
  ridgePredict,
  rowDiversityScores,
  spectralToXYZ,
  verifySubsetMatch,
  xyzToLab
};
//# sourceMappingURL=app.js.map
