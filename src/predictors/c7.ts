import { Patch, Predictor, PredictorFitInput } from '../../src/types.js';

export class C7Predictor implements Predictor {
  name = 'C7';
  private curves: { x: number[]; y: number[] }[] = [];

  fit(input: PredictorFitInput): void {
    const { anchorUnlam, anchorLam } = input;
    const k = anchorUnlam.length;
    if (k < 2) throw new Error('C7 needs at least 2 anchors');

    this.curves = [];
    for (let wl = 0; wl < 36; wl++) {
      const pairs: { a: number; b: number }[] = [];
      for (let i = 0; i < k; i++) {
        pairs.push({ a: anchorUnlam[i].spectra[wl], b: anchorLam[i].spectra[wl] });
      }
      pairs.sort((p, q) => p.a - q.a);

      const x: number[] = [];
      const y: number[] = [];
      for (const p of pairs) {
        if (x.length > 0 && p.a === x[x.length - 1]) {
          y[y.length - 1] = Math.max(y[y.length - 1], p.b);
          continue;
        }
        x.push(p.a);
        y.push(p.b);
      }

      this.enforceMonotone(x, y);
      this.curves.push({ x, y });
    }
  }

  private enforceMonotone(x: number[], y: number[]): void {
    for (let i = 1; i < y.length; i++) {
      if (y[i] < y[i - 1]) y[i] = y[i - 1];
    }
  }

  predict(patch: Patch): Float64Array {
    const out = new Float64Array(36);
    for (let wl = 0; wl < 36; wl++) {
      const { x, y } = this.curves[wl];
      const a = patch.spectra[wl];

      if (a <= x[0]) out[wl] = y[0];
      else if (a >= x[x.length - 1]) out[wl] = y[y.length - 1];
      else {
        let lo = 0, hi = x.length - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (x[mid] <= a) lo = mid;
          else hi = mid;
        }
        const t = (a - x[lo]) / (x[hi] - x[lo]);
        out[wl] = y[lo] + t * (y[hi] - y[lo]);
      }
      if (out[wl] < 0) out[wl] = 0;
      if (out[wl] > 1) out[wl] = 1;
    }
    return out;
  }
}
