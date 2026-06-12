export function spectraToXyz(spectra: Float64Array): [number, number, number] {
  let X = 0, Y = 0, Z = 0;
  const S = D50_ILLUMINANT;
  for (let i = 0; i < 36; i++) {
    const r = spectra[i];
    X += r * S[i] * CMF_X[i];
    Y += r * S[i] * CMF_Y[i];
    Z += r * S[i] * CMF_Z[i];
  }
  const k = 100 / (S.reduce((s, v, i) => s + v * CMF_Y[i], 0));
  return [X * k, Y * k, Z * k];
}

export function xyzToLab(xyz: [number, number, number], wp: [number, number, number]): [number, number, number] {
  const [x, y, z] = xyz;
  const [xw, yw, zw] = wp;
  const fx = labF(x / xw);
  const fy = labF(y / yw);
  const fz = labF(z / zw);
  const L = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const b = 200 * (fy - fz);
  return [L, a, b];
}

function labF(t: number): number {
  const delta = 6 / 29;
  return t > Math.pow(delta, 3) ? Math.cbrt(t) : t / (3 * Math.pow(delta, 2)) + 4 / 29;
}

export function deltaE00(lab1: [number, number, number], lab2: [number, number, number]): number {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;

  const avgL = (L1 + L2) / 2;
  const C1 = Math.sqrt(a1 * a1 + b1 * b1);
  const C2 = Math.sqrt(a2 * a2 + b2 * b2);
  const avgC = (C1 + C2) / 2;

  const avgC7 = Math.pow(avgC, 7);
  const G = 0.5 * (1 - Math.sqrt(avgC7 / (avgC7 + Math.pow(25, 7))));
  const a1p = a1 * (1 + G);
  const a2p = a2 * (1 + G);
  const C1p = Math.sqrt(a1p * a1p + b1 * b1);
  const C2p = Math.sqrt(a2p * a2p + b2 * b2);
  const avgCp = (C1p + C2p) / 2;

  const h1p = Math.atan2(b1, a1p) * 180 / Math.PI;
  const h2p = Math.atan2(b2, a2p) * 180 / Math.PI;
  const h1pDeg = h1p < 0 ? h1p + 360 : h1p;
  const h2pDeg = h2p < 0 ? h2p + 360 : h2p;

  const deltaLp = L2 - L1;
  const deltaCp = C2p - C1p;

  let deltahp: number;
  const diffH = h2pDeg - h1pDeg;
  if (C1p * C2p === 0) deltahp = 0;
  else if (Math.abs(diffH) <= 180) deltahp = diffH;
  else if (diffH > 180) deltahp = diffH - 360;
  else deltahp = diffH + 360;

  const deltaHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(deltahp * Math.PI / 360);

  const avgHp = (C1p * C2p === 0) ? h1pDeg + h2pDeg :
    Math.abs(h1pDeg - h2pDeg) > 180 ? (h1pDeg + h2pDeg + 360) / 2 : (h1pDeg + h2pDeg) / 2;

  const T = 1 - 0.17 * Math.cos((avgHp - 30) * Math.PI / 180)
    + 0.24 * Math.cos(2 * avgHp * Math.PI / 180)
    + 0.32 * Math.cos((3 * avgHp + 6) * Math.PI / 180)
    - 0.2 * Math.cos((4 * avgHp - 63) * Math.PI / 180);

  const SL = 1 + 0.015 * Math.pow(avgL - 50, 2) / Math.sqrt(20 + Math.pow(avgL - 50, 2));
  const SC = 1 + 0.045 * avgCp;
  const SH = 1 + 0.015 * avgCp * T;

  const deltaTheta = 30 * Math.exp(-Math.pow((avgHp - 275) / 25, 2));
  const avgCp7 = Math.pow(avgCp, 7);
  const RC = 2 * Math.sqrt(avgCp7 / (avgCp7 + Math.pow(25, 7)));
  const RT = -RC * Math.sin(2 * deltaTheta * Math.PI / 180);

  return Math.sqrt(
    Math.pow(deltaLp / SL, 2) +
    Math.pow(deltaCp / SC, 2) +
    Math.pow(deltaHp / SH, 2) +
    RT * (deltaCp / SC) * (deltaHp / SH)
  );
}

export function patchToLab(spectra: Float64Array, wp: [number, number, number]): [number, number, number] {
  return xyzToLab(spectraToXyz(spectra), wp);
}

export function computeWhitePoint(patches: { spectra: Float64Array }[]): [number, number, number] {
  for (const p of patches) {
    const xyz = spectraToXyz(p.spectra);
    if (xyz[1] > 0) return xyz;
  }
  return [96.42, 100, 82.49];
}

const D50_ILLUMINANT = new Float64Array([
  49.98, 54.65, 82.76, 91.49, 93.43, 86.68, 104.87, 117.01,
  117.81, 114.86, 115.92, 108.81, 109.35, 107.80, 104.79, 107.69,
  104.41, 104.05, 100.00, 96.33, 95.79, 88.69, 90.01, 89.60,
  87.70, 83.29, 83.70, 80.03, 80.00, 82.00, 78.00, 72.00,
  70.00, 68.00, 65.00, 64.00
]);

const CMF_X = new Float64Array([
  0.001368, 0.002236, 0.004243, 0.007650, 0.014310, 0.023190,
  0.043510, 0.077630, 0.134380, 0.214770, 0.283900, 0.328500,
  0.348280, 0.348060, 0.336200, 0.318700, 0.290800, 0.251100,
  0.195360, 0.142100, 0.095640, 0.058010, 0.032010, 0.014700,
  0.004900, 0.002400, 0.009300, 0.029100, 0.063270, 0.109600,
  0.165500, 0.225750, 0.290400, 0.359700, 0.433450, 0.512050
]);

const CMF_Y = new Float64Array([
  0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640,
  0.001210, 0.002180, 0.004000, 0.007300, 0.011600, 0.016840,
  0.023000, 0.029800, 0.038000, 0.048000, 0.060000, 0.073900,
  0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
  0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
  0.862000, 0.914850, 0.954000, 0.980300, 0.994950, 1.000000
]);

const CMF_Z = new Float64Array([
  0.006450, 0.010550, 0.020050, 0.036210, 0.067850, 0.110200,
  0.207400, 0.371300, 0.645600, 1.039050, 1.385600, 1.622960,
  1.747060, 1.782600, 1.772110, 1.744100, 1.669200, 1.528100,
  1.287640, 1.041900, 0.812950, 0.616200, 0.465180, 0.353300,
  0.272000, 0.212300, 0.158200, 0.111700, 0.078250, 0.057250,
  0.042160, 0.029840, 0.020300, 0.013400, 0.008750, 0.005750
]);
