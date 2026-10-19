import type { AEFile } from './types'

/** Dense autoencoder forward pass from the exported weights (mirrors ids_anomaly.api.app.forward). */
export function forward(ae: AEFile, x: ArrayLike<number>): Float64Array {
  let h: Float64Array = Float64Array.from(x as ArrayLike<number> as number[])
  for (const layer of ae.layers) {
    const out = new Float64Array(layer.b.length)
    for (let r = 0; r < out.length; r++) {
      const w = layer.w[r]
      let s = layer.b[r]
      for (let c = 0; c < w.length; c++) s += w[c] * h[c]
      out[r] = layer.relu && s < 0 ? 0 : s
    }
    h = out
  }
  return h
}

export interface AEScore {
  score: number
  pValue: number
  errors: Float64Array
}

/** Score a standardized feature vector; p-value from the exported calibration quantiles. */
export function scoreFlow(ae: AEFile, x: ArrayLike<number>, clip = 8): AEScore {
  const xc = Float64Array.from(Array.from(x as ArrayLike<number>, (v) => Math.max(-clip, Math.min(clip, v))))
  const recon = forward(ae, xc)
  const errors = new Float64Array(xc.length)
  let total = 0
  for (let i = 0; i < xc.length; i++) {
    const e = (recon[i] - xc[i]) ** 2
    errors[i] = e
    total += e
  }
  const score = total / xc.length
  // number of calibration quantiles >= score, rescaled to the original calibration-set size
  const cal = ae.cal_scores
  let lo = 0
  let hi = cal.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (cal[mid] < score) lo = mid + 1
    else hi = mid
  }
  const nGe = ((cal.length - lo) * ae.n_cal) / cal.length
  return { score, pValue: Math.min(1, (1 + nGe) / (ae.n_cal + 1)), errors }
}
