import type { AEFile, CurvesFile, DatasetData, DatasetMeta, Manifest, PointsFile, Replay, Summary } from './types'

const BASE = import.meta.env.BASE_URL
export const dataUrl = (path: string) => `${BASE}data/${path}`

const jsonCache = new Map<string, Promise<unknown>>()

export function fetchJson<T>(path: string): Promise<T> {
  let p = jsonCache.get(path)
  if (!p) {
    p = fetch(dataUrl(path)).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
      return r.json()
    })
    jsonCache.set(path, p)
  }
  return p as Promise<T>
}

async function fetchBuffer(path: string): Promise<ArrayBuffer> {
  const r = await fetch(dataUrl(path))
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
  return r.arrayBuffer()
}

export const loadManifest = () => fetchJson<Manifest>('manifest.json')
export const loadSummary = () => fetchJson<Summary>('summary.json')
export const loadReplay = () => fetchJson<Replay>('replay.json')

function decodeEmbedding(buf: ArrayBuffer): Float32Array {
  const raw = new Int16Array(buf)
  const out = new Float32Array(raw.length)
  for (let i = 0; i < raw.length; i++) out[i] = raw[i] / 32767
  return out
}

const datasetCache = new Map<string, Promise<DatasetData>>()

export function loadDataset(meta: DatasetMeta): Promise<DatasetData> {
  let p = datasetCache.get(meta.key)
  if (!p) {
    p = (async () => {
      const k = meta.key
      const [points, ae, curves, scores, features, attribution, ...embs] = await Promise.all([
        fetchJson<PointsFile>(`${k}/points.json`),
        fetchJson<AEFile>(`${k}/ae.json`),
        fetchJson<CurvesFile>(`${k}/curves.json`),
        fetchBuffer(`${k}/scores.u8`),
        fetchBuffer(`${k}/features.i8`),
        fetchBuffer(`${k}/attribution.u8`),
        ...meta.embeddings.map((e) => fetchBuffer(`${k}/embedding_${e}.i16`)),
      ])
      const embeddings: DatasetData['embeddings'] = {}
      meta.embeddings.forEach((e, i) => (embeddings[e] = decodeEmbedding(embs[i])))
      return {
        meta,
        n: meta.n_points,
        embeddings,
        scores: new Uint8Array(scores),
        features: new Int8Array(features),
        attack: Uint8Array.from(points.attack),
        category: Uint8Array.from(points.category),
        group: points.group ? Uint8Array.from(points.group) : null,
        groupNames: points.group_names ?? [],
        attribution: new Uint8Array(attribution),
        ae,
        curves,
      }
    })()
    datasetCache.set(meta.key, p)
  }
  return p
}

// --- accessors -------------------------------------------------------------------------------

/** -log10 conformal p-value of point ``i`` under detector index ``d``. */
export function nlogp(ds: DatasetData, d: number, i: number): number {
  return ds.scores[d * ds.n + i] / ds.meta.score_scale
}

export const pValue = (nl: number) => Math.min(1, Math.pow(10, -nl))

/** Standardized feature z-score of point ``i``, feature ``j``. */
export function featureZ(ds: DatasetData, i: number, j: number): number {
  return (ds.features[i * ds.meta.n_features + j] * ds.meta.feature_clip) / 127
}

export function topAttribution(ds: DatasetData, i: number): { feature: number; share: number }[] {
  const base = i * 10
  return Array.from({ length: 5 }, (_, r) => ({
    feature: ds.attribution[base + r],
    share: ds.attribution[base + 5 + r] / 255,
  }))
}

export function detectorIndex(ds: DatasetData, id: string): number {
  return ds.meta.detectors.findIndex((d) => d.id === id)
}
