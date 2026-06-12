import { Patch, Predictor, PredictorFitInput } from '../../src/types.js';
import { Matrix, SVD } from 'ml-matrix';

export class D1Predictor implements Predictor {
  name = 'D1';
  private paperRatio: Float64Array = new Float64Array(36);
  private pcaBasis: Matrix = new Matrix(0, 0);
  private pcaMean: Float64Array = new Float64Array(36);
  private nComponents: number = 0;
  private anchorUnlam: Patch[] = [];
  private anchorLam: Patch[] = [];

  fit(input: PredictorFitInput): void {
    const { anchorUnlam, anchorLam } = input;
    this.anchorUnlam = anchorUnlam;
    this.anchorLam = anchorLam;
    const k = anchorUnlam.length;

    const unlamPaper = anchorUnlam[0];
    const lamPaper = anchorLam[0];

    for (let wl = 0; wl < 36; wl++) {
      const unlamVal = Math.max(unlamPaper.spectra[wl], 0.001);
      const r = lamPaper.spectra[wl] / unlamVal;
      this.paperRatio[wl] = Math.max(0.3, Math.min(3.0, r));
    }

    if (k < 3) {
      this.nComponents = 0;
      return;
    }

    const residualData: number[][] = [];
    for (let i = 0; i < k; i++) {
      const row: number[] = [];
      for (let wl = 0; wl < 36; wl++) {
        const pred = this.paperRatio[wl] * anchorUnlam[i].spectra[wl];
        row.push(anchorLam[i].spectra[wl] - pred);
      }
      residualData.push(row);
    }

    this.pcaMean = new Float64Array(36);
    for (let wl = 0; wl < 36; wl++) {
      let sum = 0;
      for (let i = 0; i < k; i++) sum += residualData[i][wl];
      this.pcaMean[wl] = sum / k;
    }

    const centeredMatrix = new Matrix(residualData);
    for (let i = 0; i < k; i++) {
      for (let wl = 0; wl < 36; wl++) {
        centeredMatrix.set(i, wl, residualData[i][wl] - this.pcaMean[wl]);
      }
    }

    const svd = new SVD(centeredMatrix, { autoTranspose: true });
    this.nComponents = Math.min(4, k - 1, 36);
    const U = svd.U;
    const rows = U.length > 0 ? U.length : 0;
    if (rows > 0 && this.nComponents > 0) {
      const nRows = Math.min(rows, this.nComponents);
      this.pcaBasis = U.subMatrix(0, nRows - 1, 0, 0);
    } else {
      this.nComponents = 0;
    }
  }

  predict(patch: Patch): Float64Array {
    const out = new Float64Array(36);
    for (let wl = 0; wl < 36; wl++) {
      out[wl] = this.paperRatio[wl] * patch.spectra[wl];
    }

    if (this.nComponents > 0 && this.anchorUnlam.length >= 3) {
      const k = this.anchorUnlam.length;
      const residualVec = new Float64Array(36);
      for (let wl = 0; wl < 36; wl++) {
        const pred = this.paperRatio[wl] * patch.spectra[wl];
        residualVec[wl] = 0;
      }

      for (let comp = 0; comp < Math.min(this.nComponents, k - 1); comp++) {
        let weight = 0;
        for (let wl = 0; wl < 36; wl++) {
          weight += residualVec[wl] * this.pcaBasis.get(comp, wl);
        }
        const norm = 1;
        for (let wl = 0; wl < 36; wl++) {
          residualVec[wl] += weight * this.pcaBasis.get(comp, wl);
        }
      }

      for (let wl = 0; wl < 36; wl++) {
        out[wl] += residualVec[wl];
      }
    }

    for (let wl = 0; wl < 36; wl++) {
      if (out[wl] < 0) out[wl] = 0;
      if (out[wl] > 1) out[wl] = 1;
    }
    return out;
  }
}
