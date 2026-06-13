// Ridge regression (L2-regularized least squares)
// Browser-compatible, no external deps. Uses Cholesky decomposition.
// Solves: W = (X^T X + λI)^(-1) X^T Y

// Transpose: m×n → n×m
function transpose(A: number[][]): number[][] {
  const m = A.length, n = A[0].length;
  const AT: number[][] = Array.from({ length: n }, () => new Array(m));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      AT[j][i] = A[i][j];
  return AT;
}

// Matrix multiply: A(m×n) × B(n×p) → C(m×p)
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length, n = A[0].length, p = B[0].length;
  const C: number[][] = Array.from({ length: m }, () => new Array(p).fill(0));
  for (let i = 0; i < m; i++)
    for (let k = 0; k < n; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < p; j++)
        C[i][j] += aik * B[k][j];
    }
  return C;
}

// Cholesky decomposition: A = L L^T
// A is symmetric positive-definite n×n. Returns lower-triangular L.
function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
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

// Solve Lx = b where L is lower triangular
function solveTriangular(L: number[][], b: number[]): number[] {
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

// Solve A x = b via Cholesky: A = LL^T
// Step 1: solve L y = b for y (forward substitution)
// Step 2: solve L^T x = y for x (back substitution)
function solveCholesky(A: number[][], b: number[]): number[] {
  const L = cholesky(A);
  const n = L.length;
  // Forward: L y = b
  const y = solveTriangular(L, b);
  // Backward: L^T x = y
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++)
      sum += L[j][i] * x[j];
    x[i] = (y[i] - sum) / L[i][i];
  }
  return x;
}

export interface RidgeModel {
  weights: number[][]; // inputDims × outputDims
  lambda: number;
}

// Fit Ridge regression: X (N×p), Y (N×q), lambda
// Returns: W (p×q)
export function ridgeFit(X: number[][], Y: number[][], lambda: number = 0.01): RidgeModel {
  const N = X.length;
  const p = X[0].length;
  const q = Y[0].length;

  if (N === 0 || p === 0) throw new Error('Empty input');
  if (Y.length !== N) throw new Error('X and Y must have same number of rows');

  // X^T X (p×p)
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < p; j++) {
      const xij = X[i][j];
      if (xij === 0) continue;
      for (let k = 0; k < p; k++)
        XtX[j][k] += xij * X[i][k];
    }
  }

  // Add λI
  for (let j = 0; j < p; j++)
    XtX[j][j] += lambda;

  // X^T Y (p×q)
  const XtY: number[][] = Array.from({ length: p }, () => new Array(q).fill(0));
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < p; j++) {
      const xij = X[i][j];
      if (xij === 0) continue;
      for (let k = 0; k < q; k++)
        XtY[j][k] += xij * Y[i][k];
    }
  }

  // Solve for each output column
  const W: number[][] = Array.from({ length: p }, () => new Array(q));
  for (let k = 0; k < q; k++) {
    const b = XtY.map(row => row[k]);
    const col = solveCholesky(XtX, b);
    for (let j = 0; j < p; j++)
      W[j][k] = col[j];
  }

  return { weights: W, lambda };
}

// Predict: Y = X @ W (N×q)
export function ridgePredict(X: number[][], model: RidgeModel): number[][] {
  const Y: number[][] = [];
  for (const row of X) {
    const pred: number[] = new Array(model.weights[0].length).fill(0);
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
