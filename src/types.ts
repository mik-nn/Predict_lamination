export interface Patch {
  sampleId: string;
  cmyk: [number, number, number, number];
  spectra: Float64Array;
}

export interface DatasetPair {
  name: string;
  unlaminated: Patch[];
  laminated: Patch[];
  unlaminatedPath: string;
  laminatedPath: string;
}

export interface PredictorFitInput {
  anchorUnlam: Patch[];
  anchorLam: Patch[];
  allUnlam: Patch[];
}

export interface Predictor {
  name: string;
  fit: (input: PredictorFitInput) => void;
  predict: (patch: Patch) => Float64Array;
}

export interface S2Iteration {
  k: number;
  medianDeltaE: number;
  p95DeltaE: number;
  maxDeltaE: number;
  addedPatchId: string;
}

export interface S2Result {
  predictorName: string;
  datasetName: string;
  minK: number;
  iterations: S2Iteration[];
  finalAnchors: Patch[];
  converged: boolean;
}
