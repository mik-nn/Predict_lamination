// Color science utilities: Lab↔XYZ, D50, ΔE00 (CIEDE2000)
// All browser-compatible, no external deps

export const D50_XYZ: [number, number, number] = [0.9642, 1.0, 0.8249];
const deg2rad = Math.PI / 180;

// XYZ D50 → CIE Lab (0 ≤ L ≤ 100, -128 ≤ a,b ≤ 127)
export function xyzToLab(x: number, y: number, z: number): [number, number, number] {
  const f = (t: number) => t > 0.008856 ? Math.cbrt(t) : (903.3 * t + 16) / 116;
  const fx = f(x / D50_XYZ[0]);
  const fy = f(y);
  const fz = f(z / D50_XYZ[2]);
  const L = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const b = 200 * (fy - fz);
  return [L, a, b];
}

// CIE Lab → XYZ D50
export function labToXyz(L: number, a: number, b: number): [number, number, number] {
  const fy = (L + 16) / 116;
  const fx = a / 500 + fy;
  const fz = fy - b / 200;
  const fInv = (t: number) => {
    const t3 = t * t * t;
    return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787;
  };
  return [fInv(fx) * D50_XYZ[0], fInv(fy), fInv(fz) * D50_XYZ[2]];
}

// Spectral → XYZ (D50 illuminant, 2° observer, 380-730nm @ 10nm)
const D50_SPECTRAL: number[] = [
  49.9755, 54.6482, 82.7549, 91.486, 93.4318, 86.6823,
  104.865, 117.008, 117.812, 114.861, 115.923, 108.811,
  109.35, 107.802, 104.79, 107.689, 104.406, 97.84,
  96.334, 96.614, 92.435, 89.492, 96.259, 90.064,
  89.707, 92.244, 90.988, 89.277, 91.405, 91.015,
  89.195, 87.515, 92.155, 87.983, 84.881, 82.288,
];

const CIE_CMF_X: number[] = [
  0.000169, 0.002361, 0.01911, 0.08474, 0.2045, 0.3147,
  0.3837, 0.3707, 0.3023, 0.1956, 0.08051, 0.01617,
  0.003817, 0.01567, 0.03747, 0.04865, 0.04973, 0.04583,
  0.03801, 0.02867, 0.01984, 0.01264, 0.007417, 0.004032,
  0.002184, 0.001193, 0.000589, 0.000325, 0.000176, 0.000097,
  0.000053, 0.000028, 0.000015, 0.000009, 0.000004, 0.000002,
];
const CIE_CMF_Y: number[] = [
  0.000002, 0.000032, 0.000265, 0.001503, 0.006809, 0.01872,
  0.04844, 0.08984, 0.1282, 0.1275, 0.09146, 0.04351,
  0.01464, 0.003575, 0.001294, 0.003468, 0.007966, 0.01086,
  0.01115, 0.009887, 0.007827, 0.005646, 0.003694, 0.002148,
  0.001196, 0.000662, 0.000370, 0.000200, 0.000108, 0.000059,
  0.000032, 0.000017, 0.000010, 0.000005, 0.000003, 0.000001,
];
const CIE_CMF_Z: number[] = [
  0.000750, 0.010690, 0.08601, 0.3895, 0.9725, 1.5535,
  1.9673, 1.9948, 1.7454, 1.3176, 0.77213, 0.4149,
  0.3837, 0.3864, 0.3866, 0.3792, 0.3631, 0.3290,
  0.2795, 0.2158, 0.1571, 0.1080, 0.07098, 0.04368,
  0.02535, 0.01433, 0.008049, 0.004520, 0.002521, 0.001401,
  0.000776, 0.000434, 0.000249, 0.000137, 0.000075, 0.000040,
];

// Normalization constant for D50
const NORM_D50 = D50_SPECTRAL.reduce((s, d, i) => s + d * CIE_CMF_Y[i], 0);

export function spectralToXYZ(reflectance: number[]): [number, number, number] {
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

// CIEDE2000
export function deltaE00(L1: number, a1: number, b1: number, L2: number, a2: number, b2: number): number {
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

  const dh = Math.abs(h1pd - h2pd) > 180
    ? h2pd - h1pd + 360 * (h2pd <= h1pd ? 1 : -1)
    : h2pd - h1pd;
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin(dh * deg2rad / 2);
  const dL = L2 - L1;
  const dCp = C2p - C1p;

  const Lp_avg = (L1 + L2) / 2;
  const Cp_avg2 = (C1p + C2p) / 2;
  const hp_avg = Math.abs(h1pd - h2pd) > 180
    ? (h1pd + h2pd + 360) / 2
    : (h1pd + h2pd) / 2;

  const T = 1 - 0.17 * Math.cos((hp_avg - 30) * deg2rad)
    + 0.24 * Math.cos(2 * hp_avg * deg2rad)
    + 0.32 * Math.cos((3 * hp_avg + 6) * deg2rad)
    - 0.20 * Math.cos((4 * hp_avg - 63) * deg2rad);

  const SL = 1 + 0.015 * Math.pow(Lp_avg - 50, 2) / Math.sqrt(20 + Math.pow(Lp_avg - 50, 2));
  const SC = 1 + 0.045 * Cp_avg2;
  const SH = 1 + 0.015 * Cp_avg2 * T;

  const RC = 2 * Math.sqrt(Math.pow(Cp_avg2, 7) / (Math.pow(Cp_avg2, 7) + Math.pow(25, 7)));
  const dTheta = 30 * Math.exp(-Math.pow((hp_avg - 275) / 25, 2));
  const RT = -Math.sin(2 * dTheta * deg2rad) * RC;

  const de = Math.sqrt(
    Math.pow(dL / SL, 2) + Math.pow(dCp / SC, 2) + Math.pow(dHp / SH, 2)
    + RT * (dCp / SC) * (dHp / SH)
  );
  return de;
}
