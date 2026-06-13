// Strip matching & row recommendation for lamination DeviceLink
// CGATS parsing + row boundary computation + c-value diversity scoring
import type { Patch } from './types.ts';

export interface CGATSEntry {
  [key: string]: number;
}

export interface CGATSParseResult {
  header: Record<string, string>;
  data: CGATSEntry[];
  columns: string[];
}

// Parse CGATS file text into structured data
export function parseCGATS(text: string): CGATSParseResult {
  const lines = text.split(/\r?\n/);
  let header: Record<string, string> = {};
  let columns: string[] = [];
  let data: CGATSEntry[] = [];
  let stage: 'header' | 'format' | 'data' = 'header';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;

    if (line.startsWith('BEGIN_DATA_FORMAT')) {
      stage = 'format';
      columns = [];
      continue;
    }
    if (line.startsWith('END_DATA_FORMAT')) {
      stage = 'header';
      continue;
    }
    if (line.startsWith('BEGIN_DATA')) {
      stage = 'data';
      continue;
    }
    if (line.startsWith('END_DATA')) break;

    if (stage === 'format' && columns.length === 0) {
      columns = line.split(/\s+/);
      continue;
    }

    if (stage === 'data' && columns.length > 0) {
      const values = line.split(/\s+/);
      if (values.length >= columns.length) {
        const entry: CGATSEntry = {};
        for (let j = 0; j < columns.length; j++) {
          entry[columns[j]] = parseFloat(values[j]);
        }
        data.push(entry);
      }
      continue;
    }

    if (stage === 'header' && line.includes('\t') === false) {
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

interface RowInfo {
  start: number;
  end: number;
  count: number;
}

export function computeRows(totalPatches: number, totalRows: number): {
  rows: RowInfo[];
  patchesPerRow: number;
} {
  const patchesPerRow = Math.ceil(totalPatches / totalRows);
  const rows: RowInfo[] = [];
  let start = 0;
  for (let r = 0; r < totalRows; r++) {
    const end = Math.min(start + patchesPerRow, totalPatches);
    if (start >= totalPatches) break;
    rows.push({ start, end, count: end - start });
    start = end;
  }
  return { rows, patchesPerRow };
}

export function getRowForPatch(patchIndex: number, patchesPerRow: number): number {
  return Math.floor(patchIndex / patchesPerRow);
}

export function getPatchIndicesInRow(row: number, patchesPerRow: number, totalPatches: number): number[] {
  const start = row * patchesPerRow;
  const end = Math.min(start + patchesPerRow, totalPatches);
  const indices: number[] = [];
  for (let i = start; i < end; i++) indices.push(i);
  return indices;
}

// Extract spectral data and CMYK from CGATS
export function extractSpectralData(
  data: CGATSEntry[],
  columns: string[]
): { spectral: number[][]; cmyk: number[][]; lab: number[][] } {
  const spectralCols = columns.filter(c => /^SPECTRAL_NM_\d+$/.test(c) || /^S\d{3}$/.test(c) || c.startsWith('SPEC_'));
  const spectral: number[][] = [];
  const cmyk: number[][] = [];
  const lab: number[][] = [];

  for (const entry of data) {
    const s = spectralCols.map(c => entry[c]);
    if (s.length > 0) spectral.push(s);
    const cyan = entry['CMYK_C'] ?? entry['C'] ?? entry['CYAN'] ?? NaN;
    const magenta = entry['CMYK_M'] ?? entry['M'] ?? entry['MAGENTA'] ?? NaN;
    const yellow = entry['CMYK_Y'] ?? entry['Y'] ?? entry['YELLOW'] ?? NaN;
    const black = entry['CMYK_K'] ?? entry['K'] ?? entry['BLACK'] ?? NaN;
    if (!isNaN(cyan)) cmyk.push([cyan, magenta, yellow, black]);
    const l = entry['LAB_L'] ?? entry['L'] ?? entry['L*'] ?? NaN;
    const a = entry['LAB_A'] ?? entry['a'] ?? entry['a*'] ?? NaN;
    const b = entry['LAB_B'] ?? entry['b'] ?? entry['b*'] ?? NaN;
    if (!isNaN(l)) lab.push([l, a, b]);
  }

  return { spectral, cmyk, lab };
}

// Feature diversity score: mean pairwise Euclidean distance in feature space
// Higher diversity → better Ridge regression fit
export function featureDiversityScore(features: number[][]): number {
  const n = features.length;
  if (n < 2) return 0;
  let totalDist = 0;
  let pairs = 0;
  const step = Math.max(1, Math.floor(n / 100)); // subsample for speed
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

// Compute per-row c-value diversity scores from feature matrix
export interface RowDiversityResult {
  rowIndex: number;
  patchCount: number;
  diversity: number;
  isRecommended: boolean;
}

export function rowDiversityScores(
  features: number[][],
  patchesPerRow: number,
  totalPatches: number,
  topN?: number
): RowDiversityResult[] {
  const totalRows = Math.ceil(totalPatches / patchesPerRow);
  const results: RowDiversityResult[] = [];

  for (let r = 0; r < totalRows; r++) {
    const indices = getPatchIndicesInRow(r, patchesPerRow, totalPatches);
    if (indices.length === 0) continue;
    const rowFeatures: number[][] = [];
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

  // Sort by diversity descending, mark top N
  const sorted = [...results].sort((a, b) => b.diversity - a.diversity);
  const nMark = topN ?? 1;
  for (let i = 0; i < Math.min(nMark, sorted.length); i++) {
    const r = results.find(x => x.rowIndex === sorted[i].rowIndex);
    if (r) r.isRecommended = true;
  }

  return results;
}

// Generate a valid CGATS file containing only patches from specified rows
export function generateSubsetCGATS(
  originalText: string,
  rowIndices: number[],
  patchesPerRow: number,
  totalPatches: number
): string {
  const lines = originalText.split(/\r?\n/);

  // Find section boundaries
  let dataStart = -1;
  let dataEnd = -1;
  let numberSetsIdx = -1;

  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === 'BEGIN_DATA') dataStart = i;
    if (t === 'END_DATA' && dataStart !== -1) { dataEnd = i; break; }
    if (/^NUMBER_OF_SETS\b/.test(t)) numberSetsIdx = i;
  }

  if (dataStart === -1 || dataEnd === -1) {
    throw new Error('Could not find BEGIN_DATA / END_DATA in CGATS file');
  }

  // Collect selected patch indices
  const selectedIndices = new Set<number>();
  for (const r of rowIndices) {
    const indices = getPatchIndicesInRow(r, patchesPerRow, totalPatches);
    for (const idx of indices) selectedIndices.add(idx);
  }

  // Reconstruct CGATS
  const out: string[] = [];
  for (let i = 0; i < dataStart; i++) {
    if (i === numberSetsIdx) {
      out.push(`NUMBER_OF_SETS\t${selectedIndices.size}`);
    } else {
      out.push(lines[i]);
    }
  }

  out.push('BEGIN_DATA');
  let dataRow = 0;
  for (let i = dataStart + 1; i < dataEnd; i++) {
    const line = lines[i].trim();
    if (line.length === 0) continue;
    if (selectedIndices.has(dataRow)) {
      out.push(lines[i]);
    }
    dataRow++;
  }
  out.push('END_DATA');

  for (let i = dataEnd + 1; i < lines.length; i++) {
    out.push(lines[i]);
  }

  return out.join('\n');
}

export interface VerifyResult {
  matched: number;
  expected: number;
  missing: { sampleId: string; cmyk: number[] }[];
  extra: { sampleId: string; cmyk: number[] }[];
  ok: boolean;
}

// Verify that laminated patches match expected rows by CMYK
export function verifySubsetMatch(
  uPatches: Patch[],
  lPatches: Patch[],
  expectedRowIndices: number[],
  patchesPerRow: number
): VerifyResult {
  // Build expected CMYK multiset from U patches in selected rows
  const expectedCMYKeys = new Map<string, number>();
  for (const r of expectedRowIndices) {
    const indices = getPatchIndicesInRow(r, patchesPerRow, uPatches.length);
    for (const idx of indices) {
      const k = uPatches[idx].cmyk.join(',');
      expectedCMYKeys.set(k, (expectedCMYKeys.get(k) ?? 0) + 1);
    }
  }

  const remaining = new Map(expectedCMYKeys);
  const missing: { sampleId: string; cmyk: number[] }[] = [];
  const extra: { sampleId: string; cmyk: number[] }[] = [];
  let matched = 0;

  for (const p of lPatches) {
    const k = p.cmyk.join(',');
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
      missing.push({ sampleId: '', cmyk: k.split(',').map(Number) });
    }
  }

  const expected = lPatches.length;
  const ok = extra.length === 0 && matched === lPatches.length;

  return { matched, expected, missing, extra, ok };
}
