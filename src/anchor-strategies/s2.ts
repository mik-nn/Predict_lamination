import { Patch, Predictor, S2Result } from "../../src/types.js";
import { patchToLab, deltaE00, computeWhitePoint } from "../../src/color-math.js";

const MEDIAN_THRESHOLD = 1.0;
const P95_THRESHOLD = 2.0;

function predictFallback(predictor: Predictor, patch: Patch, anchorUnlam: Patch[], anchorLam: Patch[]): Float64Array {
  if (anchorUnlam.length >= 2) {
    return predictor.predict(patch);
  }
  if (anchorUnlam.length === 1) {
    const r = new Float64Array(36);
    for (let wl = 0; wl < 36; wl++) {
      const denom = Math.max(anchorUnlam[0].spectra[wl], 0.001);
      const ratio = anchorLam[0].spectra[wl] / denom;
      r[wl] = Math.max(0, Math.min(1, ratio * patch.spectra[wl]));
    }
    return r;
  }
  return new Float64Array(patch.spectra);
}

export function runS2(
  predictor: Predictor,
  datasetName: string,
  allUnlam: Patch[],
  allLam: Patch[],
  maxK: number = 200
): S2Result {
  const paperWhiteIdx = allLam.findIndex(
    p => p.cmyk[0] === 0 && p.cmyk[1] === 0 && p.cmyk[2] === 0 && p.cmyk[3] === 0
  );
  if (paperWhiteIdx === -1) throw new Error("No paper white patch found");

  const anchorIndices: Set<number> = new Set([paperWhiteIdx]);
  const iterations: { k: number; medianDeltaE: number; p95DeltaE: number; maxDeltaE: number; addedPatchId: string }[] = [];
  let lastAddedIdx = paperWhiteIdx;

  const wp = computeWhitePoint(allLam);

  for (let iter = 0; iter < maxK - 1; iter++) {
    const anchorUnlam = Array.from(anchorIndices).map(i => allUnlam[i]);
    const anchorLam = Array.from(anchorIndices).map(i => allLam[i]);

    if (anchorUnlam.length >= 2) {
      predictor.fit({ anchorUnlam, anchorLam, allUnlam });
    }

    const errors: { idx: number; de: number }[] = [];
    for (let i = 0; i < allUnlam.length; i++) {
      if (anchorIndices.has(i)) continue;
      const predicted = anchorUnlam.length >= 2
        ? predictor.predict(allUnlam[i])
        : predictFallback(predictor, allUnlam[i], anchorUnlam, anchorLam);
      const actualLab = patchToLab(allLam[i].spectra, wp);
      const predLab = patchToLab(predicted, wp);
      const de = deltaE00(predLab, actualLab);
      errors.push({ idx: i, de });
    }

    if (errors.length === 0) break;

    errors.sort((a, b) => a.de - b.de);
    const median = errors[Math.floor(errors.length / 2)].de;
    const p95Idx = Math.floor(errors.length * 0.95);
    const p95 = errors[Math.min(p95Idx, errors.length - 1)].de;
    const max = errors[errors.length - 1].de;

    iterations.push({
      k: anchorIndices.size,
      medianDeltaE: median,
      p95DeltaE: p95,
      maxDeltaE: max,
      addedPatchId: allLam[lastAddedIdx].sampleId,
    });

    console.log("    k=" + anchorIndices.size + " med=" + median.toFixed(3) + " p95=" + p95.toFixed(3) + " max=" + max.toFixed(3));

    if (median <= MEDIAN_THRESHOLD && p95 <= P95_THRESHOLD) {
      return {
        predictorName: predictor.name,
        datasetName,
        minK: anchorIndices.size,
        iterations,
        finalAnchors: Array.from(anchorIndices).map(i => allLam[i]),
        converged: true,
      };
    }

    const worstError = errors[errors.length - 1];
    anchorIndices.add(worstError.idx);
    lastAddedIdx = worstError.idx;
  }

  return {
    predictorName: predictor.name,
    datasetName,
    minK: anchorIndices.size,
    iterations,
    finalAnchors: Array.from(anchorIndices).map(i => allLam[i]),
    converged: false,
  };
}
