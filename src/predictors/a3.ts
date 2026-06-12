import { Patch, Predictor, PredictorFitInput } from '../../src/types.js';
import { Matrix, solve } from 'ml-matrix';

export class A3Predictor implements Predictor {
  name = 'A3';
  private slopes: Float64Array = new Float64Array(36);
  private intercepts: Float64Array = new Float64Array(36);

  fit(input: PredictorFitInput): void {
    const { anchorUnlam, anchorLam } = input;
    const k = anchorUnlam.length;
    if (k < 2) throw new Error('A3 needs at least 2 anchors');

    for (let wl = 0; wl < 36; wl++) {
      const X = new Matrix(k, 2);
      const y = new Matrix(k, 1);
      for (let i = 0; i < k; i++) {
        X.set(i, 0, anchorUnlam[i].spectra[wl]);
        X.set(i, 1, 1);
        y.set(i, 0, anchorLam[i].spectra[wl]);
      }
      const Xt = X.transpose();
      const beta = solve(Xt.mmul(X), Xt.mmul(y));
      this.slopes[wl] = beta.get(0, 0);
      this.intercepts[wl] = beta.get(1, 0);
    }
  }

  predict(patch: Patch): Float64Array {
    const out = new Float64Array(36);
    for (let wl = 0; wl < 36; wl++) {
      out[wl] = this.slopes[wl] * patch.spectra[wl] + this.intercepts[wl];
      if (out[wl] < 0) out[wl] = 0;
      if (out[wl] > 1) out[wl] = 1;
    }
    return out;
  }
}
