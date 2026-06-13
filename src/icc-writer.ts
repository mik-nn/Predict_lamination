// ICC DeviceLink profile builder (v2 lut8Type)
// Builds Lab→Lab or CMYK→CMYK DeviceLink from CLUT data

export interface DeviceLinkParams {
  inputChannels: number;   // 3 (Lab) or 4 (CMYK)
  outputChannels: number;  // 3 (Lab) or 4 (CMYK)
  clutPoints: number;      // 17 or 33 (Lab→Lab), 9 or 13 (CMYK→CMYK)
  clutData: Uint8Array;    // clutPoints^inputChannels × outputChannels
  description?: string;
}

function u16(v: number): [number, number] {
  return [(v >> 8) & 0xFF, v & 0xFF];
}
function u32(v: number): [number, number, number, number] {
  return [(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF];
}
function u8ArrayFromStrings(...strs: string[]): number[] {
  const out: number[] = [];
  for (const s of strs) for (let i = 0; i < s.length; i++) out.push(s.charCodeAt(i));
  return out;
}

// Lab↔Lab8 encoding
export function labToLab8(L: number, a: number, b: number): [number, number, number] {
  return [
    Math.round(Math.max(0, Math.min(100, L)) / 100 * 255),
    Math.round(Math.max(-128, Math.min(127, a)) + 128),
    Math.round(Math.max(-128, Math.min(127, b)) + 128),
  ];
}
export function lab8ToLab(l8: number, a8: number, b8: number): [number, number, number] {
  return [l8 / 255 * 100, a8 - 128, b8 - 128];
}

// Build an identity 1D LUT (8-bit, 256 entries)
function identityLUT(): Uint8Array {
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) lut[i] = i;
  return lut;
}

// u1Fixed15Number: 1.15 unsigned fixed point
// 1.0 = 0x8000 (32768), 0.5 = 0x4000 (16384)
function u1Fixed15(v: number): [number, number] {
  return u16(Math.round(v * 32768));
}

// Build lut8Type tag data
// Returns Uint8Array with the tag header + matrix + input LUTs + CLUT + output LUTs
function buildLUT8Tag(params: DeviceLinkParams): Uint8Array {
  const ic = params.inputChannels;
  const oc = params.outputChannels;
  const gp = params.clutPoints;
  const clut = params.clutData;

  const matrixLen = 9 * 2; // 9 × uInt16Number
  const inputLUTLen = ic * 256;
  const clutLen = Math.pow(gp, ic) * oc;
  const outputLUTLen = oc * 256;
  const tagLen = 4 + 4 + 1 + 1 + 1 + 1 + matrixLen + inputLUTLen + clutLen + outputLUTLen;

  if (clut.length !== clutLen) {
    throw new Error(`CLUT data length ${clut.length} !== expected ${clutLen} (${gp}^${ic} × ${oc})`);
  }

  const buf = new Uint8Array(tagLen);
  let off = 0;

  // 'mft1' signature
  buf.set([0x6D, 0x66, 0x74, 0x31], off); off += 4;
  // Reserved
  buf.set([0, 0, 0, 0], off); off += 4;
  // Input channels, output channels, CLUT points
  buf[off++] = ic;
  buf[off++] = oc;
  buf[off++] = gp;
  buf[off++] = 0; // reserved

  // Identity matrix: [1 0 0; 0 1 0; 0 0 1] in u1Fixed15
  // For input channels > 3, matrix only applies to first 3 channels
  if (ic >= 1 && ic <= 15) {
    const ident3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (const v of ident3) {
      buf.set(u1Fixed15(v), off); off += 2;
    }
  } else {
    for (let i = 0; i < 9; i++) { buf.set(u16(0), off); off += 2; }
  }

  // Input LUTs (identity)
  for (let c = 0; c < ic; c++) {
    const lut = identityLUT();
    buf.set(lut, off); off += 256;
  }

  // CLUT
  buf.set(clut, off); off += clutLen;

  // Output LUTs (identity)
  for (let c = 0; c < oc; c++) {
    const lut = identityLUT();
    buf.set(lut, off); off += 256;
  }

  return buf;
}

// Build a complete ICC DeviceLink profile
export function buildDeviceLink(params: DeviceLinkParams): ArrayBuffer {
  const desc = params.description || 'Unlaminated→Laminated DeviceLink';
  const ic = params.inputChannels;
  const oc = params.outputChannels;

  // 1. Build AToB0 tag
  const a2b0Tag = buildLUT8Tag(params);

  // 2. Build header (128 bytes)
  const tagCount = 1;
  const tagTableStart = 128;
  const tagTableLen = 4 + tagCount * 12; // count + tag entries
  const a2b0Offset = tagTableStart + tagTableLen;
  const profileSize = a2b0Offset + a2b0Tag.length;

  const header = new Uint8Array(128);
  let off = 0;

  // Profile size
  header.set(u32(profileSize), off); off += 4;
  // CMM: 'none'
  header.set(u8ArrayFromStrings('none'), off); off += 4;
  // Version: 2.4.0 (0x02400000)
  header.set([0x02, 0x40, 0x00, 0x00], off); off += 4;
  // Device class: 'link' (DeviceLink)
  header.set(u8ArrayFromStrings('link'), off); off += 4;
  // Color space
  header.set(u8ArrayFromStrings(ic === 4 ? 'CMYK' : 'Lab '), off); off += 4;
  // PCS
  header.set(u8ArrayFromStrings(oc === 4 ? 'CMYK' : 'Lab '), off); off += 4;
  // Date/time (Jan 1 2024 00:00:00)
  header.set(u16(2024), off); off += 2; // year
  header.set(u16(1), off); off += 2;    // month
  header.set(u16(1), off); off += 2;    // day
  header.set(u16(0), off); off += 2;    // hour
  header.set(u16(0), off); off += 2;    // minute
  header.set(u16(0), off); off += 2;    // second
  // 'acsp' signature
  header.set(u8ArrayFromStrings('acsp'), off); off += 4;
  // Platform: 'APPL'
  header.set(u8ArrayFromStrings('APPL'), off); off += 4;
  // Flags: 0
  header.set(u32(0), off); off += 4;
  // Device manufacturer: 0
  header.set(u32(0), off); off += 4;
  // Device model: 0
  header.set(u32(0), off); off += 4;
  // Device attributes: 0
  header.set(u32(0), off); off += 4;
  header.set(u32(0), off); off += 4;
  // Rendering intent: 0 (Perceptual)
  header.set(u32(0), off); off += 4;
  // PCS illuminant: D50 XYZ (96.42, 100.00, 82.49 in s15Fixed16)
  // s15Fixed16: X*2^16, Y*2^16, Z*2^16
  const d50X = Math.round(0.9642 * 65536);
  const d50Y = Math.round(1.0000 * 65536);
  const d50Z = Math.round(0.8249 * 65536);
  header.set(u32(d50X), off); off += 4;
  header.set(u32(d50Y), off); off += 4;
  header.set(u32(d50Z), off); off += 4;
  // Creator: 'none'
  header.set(u8ArrayFromStrings('none'), off); off += 4;
  // Reserved (44 bytes of zeros)
  for (let i = 0; i < 44; i++) header[off++] = 0;

  // 3. Tag table
  const tagTable = new Uint8Array(tagTableLen);
  let toff = 0;
  tagTable.set(u32(tagCount), toff); toff += 4;
  // AToB0 tag
  tagTable.set(u8ArrayFromStrings('A2B0'), toff); toff += 4;
  tagTable.set(u32(a2b0Offset), toff); toff += 4;
  tagTable.set(u32(a2b0Tag.length), toff); toff += 4;

  // Description tag (optional but recommended)
  // For now, skip to keep it simple. Most color engines don't require it.

  // 4. Combine
  const profile = new Uint8Array(profileSize);
  profile.set(header, 0);
  profile.set(tagTable, tagTableStart);
  profile.set(a2b0Tag, a2b0Offset);

  return profile.buffer as ArrayBuffer;
}

// Helper: build a uniform CLUT grid and fill it
// For Lab→Lab: grid of (clutPoints)^3 Lab8 values
// For CMYK→CMYK: grid of (clutPoints)^4 uint8 values
export function allocateCLUT(inputChannels: number, outputChannels: number, clutPoints: number): {
  array: Uint8Array;
  setter: (indices: number[], values: number[]) => void;
  getter: (indices: number[]) => number[];
} {
  const size = Math.pow(clutPoints, inputChannels) * outputChannels;
  const array = new Uint8Array(size);

  function linearIndex(indices: number[]): number {
    let idx = 0;
    for (let c = 0; c < inputChannels; c++) {
      idx = idx * clutPoints + indices[c];
    }
    return idx * outputChannels;
  }

  return {
    array,
    setter: (indices: number[], values: number[]) => {
      const base = linearIndex(indices);
      for (let k = 0; k < outputChannels; k++) array[base + k] = values[k];
    },
    getter: (indices: number[]) => {
      const base = linearIndex(indices);
      const out: number[] = [];
      for (let k = 0; k < outputChannels; k++) out.push(array[base + k]);
      return out;
    }
  };
}

// ---- DeviceLink CLUT interpolation ----

// Apply Lab→Lab DeviceLink CLUT via trilinear interpolation
// labIn: [L, a, b] in standard Lab space (L 0-100, a/b -128..127)
// clutData: Uint8Array from allocateCLUT (gp^3 × 3, Lab8 encoded)
// clutPoints: grid size (17 or 33)
// Returns: [L, a, b] after DeviceLink transform
export function applyCLUT(
  labIn: [number, number, number],
  clutData: Uint8Array,
  clutPoints: number
): [number, number, number] {
  const [L, a, b] = labIn;
  const oc = 3;

  // Normalize input to grid coordinates [0, gp-1]
  const lf = (L / 100) * (clutPoints - 1);
  const af = ((a + 128) / 255) * (clutPoints - 1);
  const bf = ((b + 128) / 255) * (clutPoints - 1);

  const l0 = Math.max(0, Math.min(clutPoints - 2, Math.floor(lf)));
  const a0 = Math.max(0, Math.min(clutPoints - 2, Math.floor(af)));
  const b0 = Math.max(0, Math.min(clutPoints - 2, Math.floor(bf)));

  const l1 = l0 + 1;
  const a1 = a0 + 1;
  const b1 = b0 + 1;

  const ld = lf - l0;
  const ad = af - a0;
  const bd = bf - b0;

  function grid(l: number, a: number, b: number): [number, number, number] {
    const idx = ((l * clutPoints + a) * clutPoints + b) * oc;
    return [clutData[idx] / 255 * 100, clutData[idx + 1] - 128, clutData[idx + 2] - 128];
  }

  const v000 = grid(l0, a0, b0);
  const v100 = grid(l1, a0, b0);
  const v010 = grid(l0, a1, b0);
  const v110 = grid(l1, a1, b0);
  const v001 = grid(l0, a0, b1);
  const v101 = grid(l1, a0, b1);
  const v011 = grid(l0, a1, b1);
  const v111 = grid(l1, a1, b1);

  function lerp(c00: number[], c10: number[], c01: number[], c11: number[], dx: number, dy: number, dz: number): number {
    const c0 = c00[dz] * (1 - dx) + c10[dz] * dx;
    const c1 = c01[dz] * (1 - dx) + c11[dz] * dx;
    return c0 * (1 - dy) + c1 * dy;
  }

  const v0 = [v000, v100, v010, v110];
  const v1 = [v001, v101, v011, v111];

  const Lout = lerp(v000, v100, v010, v110, ld, ad, 0) * (1 - bd) + lerp(v001, v101, v011, v111, ld, ad, 0) * bd;
  const aout = lerp(v000, v100, v010, v110, ld, ad, 1) * (1 - bd) + lerp(v001, v101, v011, v111, ld, ad, 1) * bd;
  const bout = lerp(v000, v100, v010, v110, ld, ad, 2) * (1 - bd) + lerp(v001, v101, v011, v111, ld, ad, 2) * bd;

  return [Lout, aout, bout];
}

// Convert hex string to ArrayBuffer (for downloading)
export function downloadICC(buffer: ArrayBuffer, filename: string) {
  const blob = new Blob([buffer], { type: 'application/vnd.iccprofile' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
