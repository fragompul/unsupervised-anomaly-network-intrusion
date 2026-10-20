import { useEffect, useRef, useState } from 'react'
import { loadDataset, loadManifest, loadSummary } from './data'
import type { DatasetData, Manifest, Summary } from './types'

/** Generic async loader with loading / error state. */
export function useAsync<T>(factory: () => Promise<T>, deps: unknown[]): { data: T | null; error: Error | null; loading: boolean } {
  const [state, setState] = useState<{ data: T | null; error: Error | null; loading: boolean }>({ data: null, error: null, loading: true })
  useEffect(() => {
    let live = true
    setState((s) => ({ ...s, loading: true, error: null }))
    factory().then(
      (data) => live && setState({ data, error: null, loading: false }),
      (error) => live && setState({ data: null, error, loading: false }),
    )
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return state
}

export const useManifest = () => useAsync<Manifest>(loadManifest, [])
export const useSummary = () => useAsync<Summary>(loadSummary, [])

export function useDataset(manifest: Manifest | null, key: string) {
  return useAsync<DatasetData | null>(
    async () => {
      const meta = manifest?.datasets.find((d) => d.key === key)
      return meta ? loadDataset(meta) : null
    },
    [manifest, key],
  )
}

export function useSize<T extends HTMLElement>(): [React.RefObject<T | null>, { w: number; h: number }] {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.round(e.contentRect.width), h: Math.round(e.contentRect.height) }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size]
}

/** Animated count-up for headline numbers. */
export function useCountUp(target: number, ms = 1100): number {
  const [v, setV] = useState(0)
  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) {
      setV(target)
      return
    }
    let raf = 0
    const t0 = performance.now()
    const tick = (t: number) => {
      const f = Math.min(1, (t - t0) / ms)
      setV(target * (1 - Math.pow(1 - f, 3)))
      if (f < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}
