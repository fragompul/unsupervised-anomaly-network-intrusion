import { describe, expect, it } from 'vitest'
import { forward, scoreFlow } from './ae'
import type { AEFile } from './types'

const ae: AEFile = {
  layers: [
    { w: [[1, 0], [0, 2]], b: [0, -1], relu: true },
    { w: [[1, 1], [0, 1]], b: [0.5, 0], relu: false },
  ],
  cal_scores: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  n_cal: 1000,
  median_error_per_feature: [0, 0],
}

describe('autoencoder forward pass', () => {
  it('matches a hand-computed two-layer network (same case as the Python test)', () => {
    // h1 = relu([2, 6-1]) = [2, 5]; out = [2+5+0.5, 5]
    expect(Array.from(forward(ae, [2, 3]))).toEqual([7.5, 5])
  })

  it('applies ReLU only where flagged', () => {
    const out = forward({ ...ae, layers: [{ w: [[1, 0]], b: [0], relu: true }] }, [-3, 0])
    expect(out[0]).toBe(0)
  })
})

describe('scoreFlow', () => {
  it('gives a smaller p-value to a more anomalous flow', () => {
    const near = scoreFlow(ae, [0.1, 0.1])
    const far = scoreFlow(ae, [7, 7])
    expect(far.score).toBeGreaterThan(near.score)
    expect(far.pValue).toBeLessThan(near.pValue)
    expect(far.pValue).toBeGreaterThan(0)
  })

  it('p-value is 1 for a flow below every calibration score and respects the (1+n)/(N+1) floor', () => {
    const tiny = scoreFlow({ ...ae, layers: [{ w: [[1, 0], [0, 1]], b: [0, 0], relu: false }] }, [0, 0])
    expect(tiny.score).toBe(0)
    expect(tiny.pValue).toBe(1)
    const huge = scoreFlow(ae, [8, 8])
    expect(huge.pValue).toBeCloseTo(1 / 1001, 6)
  })

  it('clips inputs to the feature range like the Python reference', () => {
    expect(scoreFlow(ae, [100, 100]).score).toBeCloseTo(scoreFlow(ae, [8, 8]).score, 10)
  })
})
