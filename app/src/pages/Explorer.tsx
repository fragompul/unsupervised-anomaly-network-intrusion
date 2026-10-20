import { AnimatePresence, motion } from 'framer-motion'
import { Crosshair, Pause, Play, X } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { FlowInspector } from '../components/FlowInspector'
import { PointCloud } from '../components/three/PointCloud'
import { Badge, Segmented, Select, Skeleton, Slider } from '../components/ui'
import { nlogp, pValue } from '../lib/data'
import { DATASET_SHORT } from '../lib/format'
import { useData } from '../lib/context'
import { useDataset } from '../lib/hooks'
import { CATEGORY_LABEL, categoryColor } from '../lib/palette'
import { useExplorer } from '../lib/store'
import { fmt } from '../lib/format'

export default function Explorer() {
  const { manifest } = useData()
  const s = useExplorer()
  useEffect(() => {
    if (manifest && !manifest.datasets.some((d) => d.key === s.datasetKey) && manifest.datasets[0]) s.set({ datasetKey: manifest.datasets[0].key })
  }, [manifest, s])
  const { data: ds, loading } = useDataset(manifest, s.datasetKey)
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null)

  const onHover = useCallback((i: number | null, x: number, y: number) => setHover(i == null ? null : { i, x, y }), [])
  const onSelect = useCallback((i: number | null) => s.set({ selected: i }), [s])

  const detIdx = useMemo(() => Math.max(0, ds?.meta.detectors.findIndex((d) => d.id === s.detector) ?? 0), [ds, s.detector])

  const stats = useMemo(() => {
    if (!ds) return null
    let tp = 0, fp = 0, fn = 0, tn = 0
    for (let i = 0; i < ds.n; i++) {
      const flagged = pValue(nlogp(ds, detIdx, i)) <= s.alpha
      const attack = ds.attack[i] === 1
      if (flagged && attack) tp++
      else if (flagged) fp++
      else if (attack) fn++
      else tn++
    }
    return { tp, fp, fn, tn, tpr: tp / Math.max(1, tp + fn), fpr: fp / Math.max(1, fp + tn), precision: tp / Math.max(1, tp + fp), flagged: tp + fp }
  }, [ds, detIdx, s.alpha])

  const top = useMemo(() => {
    if (!ds) return []
    const rows: { i: number; nl: number }[] = []
    for (let i = 0; i < ds.n; i++) rows.push({ i, nl: nlogp(ds, detIdx, i) })
    return rows.sort((a, b) => b.nl - a.nl).slice(0, 40)
  }, [ds, detIdx])

  const catCounts = useMemo(() => {
    if (!ds) return []
    const c = new Array(ds.meta.categories.length).fill(0)
    for (let i = 0; i < ds.n; i++) c[ds.category[i]]++
    return c as number[]
  }, [ds])

  const detOptions = useMemo(
    () => (ds?.meta.detectors ?? []).map((d) => ({ value: d.id, label: d.title, group: d.family === 'ensemble' ? 'Ensembles' : 'Detectors' })),
    [ds],
  )

  return (
    <div className="relative h-screen min-h-[640px] pt-14">
      {/* scene */}
      <div className="absolute inset-0 top-0">
        {ds && manifest ? (
          <PointCloud
            className="h-full w-full"
            ds={ds}
            embedding={ds.embeddings[s.embedding] ? s.embedding : 'pca'}
            colorBy={s.colorBy}
            detectorIndex={detIdx}
            alpha={s.alpha}
            hiddenCategories={s.hiddenCategories}
            selected={s.selected}
            onSelect={onSelect}
            onHover={onHover}
            autoRotate={s.autoRotate && s.selected == null}
          />
        ) : (
          <div className="flex h-full items-center justify-center"><Skeleton className="h-64 w-64 rounded-full" /></div>
        )}
      </div>
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,rgba(5,7,13,0.65)_100%)]" />

      {/* left panel */}
      <aside className="glass absolute bottom-4 left-4 top-[72px] z-10 w-[308px] overflow-y-auto p-4 max-lg:hidden">
        <div className="space-y-5">
          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-faint">Dataset</div>
            <Segmented
              label="dataset"
              size="sm"
              value={s.datasetKey}
              onChange={(v) => s.set({ datasetKey: v, selected: null, hiddenCategories: [] })}
              options={(manifest?.datasets ?? []).map((d) => ({ value: d.key, label: DATASET_SHORT[d.key] ?? d.title }))}
            />
            {ds && <p className="mt-2 text-xs leading-relaxed text-muted">{fmt.int(ds.meta.n_test)} test flows · {ds.n.toLocaleString()} shown (rare classes kept whole) · {ds.meta.n_features} features</p>}
          </div>

          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-faint">Embedding</div>
            <Segmented
              label="embedding"
              size="sm"
              value={s.embedding}
              onChange={(v) => s.set({ embedding: v })}
              options={[
                { value: 'pca', label: 'PCA (linear)' },
                { value: 'umap', label: 'UMAP (manifold)', disabled: !ds?.embeddings.umap },
              ]}
            />
          </div>

          <Select label="Detector" value={s.detector} onChange={(v) => s.set({ detector: v })} options={detOptions} />

          <div>
            <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-faint">Colour by</div>
            <Segmented
              label="colorby"
              size="sm"
              value={s.colorBy}
              onChange={(v) => s.set({ colorBy: v })}
              options={[
                { value: 'category', label: 'Category' },
                { value: 'score', label: 'Evidence' },
                { value: 'truth', label: 'Truth' },
              ]}
            />
          </div>

          <Slider label="Flag at α (false-positive budget)" value={Math.log10(s.alpha)} min={-3} max={-0.7} step={0.01} display={`${(s.alpha * 100).toFixed(s.alpha < 0.01 ? 2 : 1)}%`} onChange={(v) => s.set({ alpha: Math.pow(10, v) })} />

          {stats && (
            <div className="grid grid-cols-3 gap-2 rounded-xl border border-line bg-white/[0.025] p-3">
              <MiniStat label="Flagged" value={fmt.int(stats.flagged)} />
              <MiniStat label="Recall" value={fmt.pct(stats.tpr, 0)} color="#34d399" />
              <MiniStat label="FPR" value={fmt.pct(stats.fpr, 1)} color={stats.fpr > s.alpha * 1.5 ? '#fb7185' : '#e8ebf5'} />
            </div>
          )}

          {ds && s.colorBy === 'category' && (
            <div>
              <div className="mb-1.5 flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-faint">
                Categories
                {s.hiddenCategories.length > 0 && <button className="text-violet-soft hover:underline" onClick={() => s.set({ hiddenCategories: [] })}>show all</button>}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {ds.meta.categories.map((c, i) => {
                  const off = s.hiddenCategories.includes(c)
                  return (
                    <button
                      key={c}
                      onClick={() => s.toggleCategory(c)}
                      className="focus-ring flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs transition-all hover:bg-white/[0.06]"
                      style={{ opacity: off ? 0.35 : 1 }}
                    >
                      <span className="h-2 w-2 rounded-full" style={{ background: categoryColor(c, i), boxShadow: `0 0 8px ${categoryColor(c, i)}` }} />
                      {CATEGORY_LABEL(c)}
                      <span className="num text-faint">{catCounts[i]}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          )}
          {s.colorBy === 'score' && (
            <div>
              <div className="h-2 rounded-full" style={{ background: 'linear-gradient(90deg, rgb(38,48,92), rgb(112,94,255), rgb(236,90,190), rgb(255,214,102))' }} />
              <div className="num mt-1 flex justify-between text-[10px] text-faint"><span>p = 1</span><span>10⁻²</span><span>10⁻⁵</span></div>
            </div>
          )}
        </div>
      </aside>

      {/* floating controls */}
      <div className="absolute bottom-6 left-1/2 z-10 flex -translate-x-1/2 items-center gap-2 max-lg:left-4 max-lg:translate-x-0">
        <button onClick={() => s.set({ autoRotate: !s.autoRotate })} className="focus-ring glass-flat flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-muted hover:text-text" style={{ background: 'rgba(8,11,20,0.7)' }}>
          {s.autoRotate ? <Pause size={13} /> : <Play size={13} />} {s.autoRotate ? 'Pause rotation' : 'Rotate'}
        </button>
        <div className="glass-flat px-3 py-1.5 text-xs text-faint" style={{ background: 'rgba(8,11,20,0.7)' }}>drag · scroll · click a point</div>
      </div>

      {/* right panel */}
      <aside className="glass absolute bottom-4 right-4 top-[72px] z-10 w-[348px] overflow-y-auto p-4 max-lg:hidden">
        <AnimatePresence mode="wait">
          {ds && s.selected != null ? (
            <motion.div key={`sel-${s.selected}`} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}>
              <div className="mb-4 flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-[15px] font-semibold"><Crosshair size={15} className="text-violet-soft" /> Flow #{s.selected}</h2>
                <button className="focus-ring rounded-lg p-1 text-faint hover:bg-white/[0.07] hover:text-text" onClick={() => s.set({ selected: null })} aria-label="Close inspector"><X size={16} /></button>
              </div>
              <FlowInspector ds={ds} index={s.selected} detector={s.detector} alpha={s.alpha} />
            </motion.div>
          ) : (
            <motion.div key="top" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <h2 className="text-[15px] font-semibold">Top anomalies</h2>
              <p className="mb-3 mt-1 text-xs text-muted">Ranked by {ds?.meta.detectors[detIdx]?.title}. Click a row, or a point in the scene, to open the inspector.</p>
              <div className="space-y-1">
                {ds && top.map(({ i, nl }) => {
                  const c = ds.meta.categories[ds.category[i]]
                  return (
                    <button key={i} onClick={() => s.set({ selected: i })} className="focus-ring group flex w-full items-center gap-3 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-white/[0.06]">
                      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: categoryColor(c, ds.category[i]) }} />
                      <span className="num w-12 text-muted">#{i}</span>
                      <span className="flex-1 truncate text-text">{CATEGORY_LABEL(c)}</span>
                      <span className="num text-violet-soft">{nl.toFixed(1)}</span>
                    </button>
                  )
                })}
                {loading && <Skeleton className="h-40" />}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </aside>

      {/* hover tooltip */}
      {hover && ds && (
        <div className="pointer-events-none fixed z-50 -translate-y-full rounded-lg border border-line bg-[#0b101c]/95 px-3 py-2 text-xs shadow-xl backdrop-blur" style={{ left: hover.x + 14, top: hover.y - 10 }}>
          <div className="flex items-center gap-2 font-medium">
            <span className="h-2 w-2 rounded-full" style={{ background: categoryColor(ds.meta.categories[ds.category[hover.i]], ds.category[hover.i]) }} />
            {CATEGORY_LABEL(ds.meta.categories[ds.category[hover.i]])}
          </div>
          <div className="num mt-0.5 text-muted">#{hover.i} · p = {fmt.p(pValue(nlogp(ds, detIdx, hover.i)))}</div>
        </div>
      )}

      {/* small-screen notice */}
      <div className="absolute inset-x-4 top-[104px] z-10 md:top-[72px] lg:hidden">
        <div className="glass p-4 text-sm text-muted">
          The explorer’s control panels are designed for larger screens. <Badge color="#8b7bff">Tip</Badge> Use the 3D view directly, or open this page on desktop.
        </div>
      </div>
    </div>
  )
}

function MiniStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-faint">{label}</div>
      <div className="num mt-0.5 text-base font-medium" style={{ color }}>{value}</div>
    </div>
  )
}
