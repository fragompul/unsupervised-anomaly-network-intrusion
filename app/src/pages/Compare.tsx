import { useEffect, useMemo, useState } from 'react'
import { LineChart } from '../components/charts/LineChart'
import { PointCloud } from '../components/three/PointCloud'
import { Card, Empty, PageHeader, SectionTitle, Segmented, Select, Skeleton, Slider } from '../components/ui'
import { useData } from '../lib/context'
import { nlogp, pValue } from '../lib/data'
import { DATASET_SHORT, fmt } from '../lib/format'
import { useDataset } from '../lib/hooks'
import { CATEGORY_LABEL, categoryColor } from '../lib/palette'

const A_COLOR = '#8b7bff'
const B_COLOR = '#22d3ee'

export default function Compare() {
  const { manifest } = useData()
  const [key, setKey] = useState('')
  useEffect(() => {
    if (manifest && !key) setKey(manifest.datasets.find((d) => d.key === 'unsw_nb15')?.key ?? manifest.datasets[0]?.key ?? '')
  }, [manifest, key])
  const { data: ds } = useDataset(manifest, key)
  const [a, setA] = useState('ae')
  const [b, setB] = useState('iforest')
  const [alpha, setAlpha] = useState(0.05)
  const [curve, setCurve] = useState<'roc' | 'pr'>('roc')

  const ia = ds?.meta.detectors.findIndex((d) => d.id === a) ?? -1
  const ib = ds?.meta.detectors.findIndex((d) => d.id === b) ?? -1
  const options = (ds?.meta.detectors ?? []).map((d) => ({ value: d.id, label: d.title }))
  const title = (id: string) => ds?.meta.detectors.find((d) => d.id === id)?.title ?? id

  const agreement = useMemo(() => {
    if (!ds || ia < 0 || ib < 0) return null
    const cell = { both: [0, 0], onlyA: [0, 0], onlyB: [0, 0], none: [0, 0] } // [normal, attack]
    const perCat = ds.meta.categories.map(() => ({ n: 0, a: 0, b: 0, onlyA: 0, onlyB: 0 }))
    for (let i = 0; i < ds.n; i++) {
      const fa = pValue(nlogp(ds, ia, i)) <= alpha
      const fb = pValue(nlogp(ds, ib, i)) <= alpha
      const t = ds.attack[i]
      cell[fa && fb ? 'both' : fa ? 'onlyA' : fb ? 'onlyB' : 'none'][t]++
      const c = perCat[ds.category[i]]
      c.n++
      if (fa) c.a++
      if (fb) c.b++
      if (fa && !fb) c.onlyA++
      if (fb && !fa) c.onlyB++
    }
    return { cell, perCat }
  }, [ds, ia, ib, alpha])

  if (!manifest) return <Shell><Skeleton className="h-96" /></Shell>
  if (!ds || ia < 0 || ib < 0) return <Shell><Skeleton className="h-96" /></Shell>

  const ca = ds.curves.curves[a]
  const cb = ds.curves.curves[b]
  const series = (c: typeof ca, id: string, color: string) => ({
    id, label: title(id), color,
    points: (curve === 'roc' ? c.fpr.map((x, i) => [x, c.tpr[i]]) : c.recall.map((x, i) => [x, c.precision[i]])) as [number, number][],
  })
  const total = agreement ? Object.values(agreement.cell).reduce((s, v) => s + v[0] + v[1], 0) : 1

  return (
    <Shell>
      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_1fr_1fr_1.1fr]">
        <Card className="p-4"><Select label="Dataset" value={key} onChange={setKey} options={manifest.datasets.map((d) => ({ value: d.key, label: DATASET_SHORT[d.key] ?? d.title }))} /></Card>
        <Card className="p-4" style={{ boxShadow: `inset 3px 0 0 ${A_COLOR}` }}><Select label="Detector A" value={a} onChange={setA} options={options} /></Card>
        <Card className="p-4" style={{ boxShadow: `inset 3px 0 0 ${B_COLOR}` }}><Select label="Detector B" value={b} onChange={setB} options={options} /></Card>
        <Card className="p-4"><Slider label="Flag at α" value={Math.log10(alpha)} min={-3} max={-0.7} step={0.01} display={`${(alpha * 100).toFixed(alpha < 0.01 ? 2 : 1)}%`} onChange={(v) => setAlpha(Math.pow(10, v))} /></Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_1.1fr]">
        <Card className="p-6">
          <div className="mb-2 flex items-center justify-between">
            <SectionTitle>{curve === 'roc' ? 'ROC curve' : 'Precision–recall'} · full test set</SectionTitle>
            <Segmented size="sm" label="curve" value={curve} onChange={setCurve} options={[{ value: 'roc', label: 'ROC' }, { value: 'pr', label: 'PR' }]} />
          </div>
          <LineChart
            height={330}
            series={[series(ca, a, A_COLOR), series(cb, b, B_COLOR)]}
            xScale={curve === 'roc' ? 'log' : 'linear'}
            xDomain={curve === 'roc' ? [1e-4, 1] : [0, 1]}
            yDomain={[0, 1]}
            xLabel={curve === 'roc' ? 'False-positive rate (log)' : 'Recall'}
            yLabel={curve === 'roc' ? 'True-positive rate' : 'Precision'}
            formatX={(v) => (curve === 'roc' ? (v >= 0.1 ? String(v) : v.toExponential(0)) : v.toFixed(1))}
            formatY={(v) => v.toFixed(1)}
            diagonal={curve === 'roc' ? false : false}
            hoverX
          />
          <div className="mt-3 flex gap-6 text-[13px]">
            <Legend color={A_COLOR} label={title(a)} />
            <Legend color={B_COLOR} label={title(b)} />
          </div>
        </Card>

        <Card className="relative min-h-[420px] overflow-hidden p-0">
          <div className="absolute left-5 top-5 z-10">
            <h2 className="text-[15px] font-semibold">Disagreement map</h2>
            <p className="mt-1 max-w-[300px] text-xs text-muted">Each dot is a flow. <span style={{ color: '#b6acff' }}>Violet</span>: only A flags it · <span style={{ color: B_COLOR }}>cyan</span>: only B · <span className="text-white">white</span>: both.</p>
          </div>
          <PointCloud className="absolute inset-0" ds={ds} embedding={ds.embeddings.umap ? 'umap' : 'pca'} colorBy="diff" detectorIndex={ia} detectorIndexB={ib} alpha={alpha} hiddenCategories={[]} selected={null} bloom autoRotate rotateSpeed={0.6} onSelect={undefined} />
        </Card>
      </div>

      {agreement && (
        <div className="mt-6 grid gap-6 xl:grid-cols-[1fr_1.4fr]">
          <Card className="p-6">
            <SectionTitle hint={`${fmt.int(ds.n)} sampled flows`}>Who flags what</SectionTitle>
            <div className="mt-3 grid grid-cols-2 gap-3">
              {([['both', 'Both flag', '#ffffff'], ['onlyA', `Only ${title(a)}`, A_COLOR], ['onlyB', `Only ${title(b)}`, B_COLOR], ['none', 'Neither', '#5d6684']] as const).map(([k, label, color]) => {
                const [nor, att] = agreement.cell[k]
                return (
                  <div key={k} className="rounded-xl border border-line bg-white/[0.025] p-4">
                    <div className="flex items-center gap-2 text-xs text-muted"><span className="h-2 w-2 rounded-full" style={{ background: color }} /><span className="truncate">{label}</span></div>
                    <div className="num mt-2 text-2xl">{fmt.int(nor + att)}</div>
                    <div className="mt-1 flex gap-3 text-xs"><span className="text-rose">{fmt.int(att)} attacks</span><span className="text-mint">{fmt.int(nor)} normal</span></div>
                    <div className="mt-2 h-1 rounded-full bg-white/[0.06]"><div className="h-full rounded-full" style={{ width: `${((nor + att) / total) * 100}%`, background: color }} /></div>
                  </div>
                )
              })}
            </div>
          </Card>
          <Card className="p-6">
            <SectionTitle hint="Recall by attack category at the chosen α">Per-category recall</SectionTitle>
            <div className="mt-3 space-y-3">
              {agreement.perCat.map((c, i) => {
                const cat = ds.meta.categories[i]
                return (
                  <div key={cat} className="grid grid-cols-[110px_1fr_92px] items-center gap-3 text-[13px]">
                    <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full" style={{ background: categoryColor(cat, i) }} />{CATEGORY_LABEL(cat)}</span>
                    <div className="space-y-1">
                      <Bar v={c.a / Math.max(1, c.n)} color={A_COLOR} />
                      <Bar v={c.b / Math.max(1, c.n)} color={B_COLOR} />
                    </div>
                    <span className="num text-right text-xs text-muted">{cat === 'normal' ? 'FPR ' : ''}{fmt.pct(c.a / Math.max(1, c.n), 0)} · {fmt.pct(c.b / Math.max(1, c.n), 0)}</span>
                  </div>
                )
              })}
            </div>
          </Card>
        </div>
      )}
      {!agreement && <Empty>Select two detectors.</Empty>}
    </Shell>
  )
}

function Bar({ v, color }: { v: number; color: string }) {
  return <div className="h-1.5 rounded-full bg-white/[0.06]"><div className="h-full rounded-full transition-all duration-500" style={{ width: `${v * 100}%`, background: color }} /></div>
}
function Legend({ color, label }: { color: string; label: string }) {
  return <span className="flex items-center gap-2 text-muted"><span className="h-[3px] w-5 rounded" style={{ background: color }} />{label}</span>
}
function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-[1500px] px-6 pb-24 pt-28">
      <PageHeader eyebrow="Compare" title={<>Two detectors, <span className="grad-text">one disagreement map.</span></>}>
        Aggregate AUC hides the interesting part: which flows does one method catch that the other misses? Pick any pair and a false-positive budget.
      </PageHeader>
      {children}
    </div>
  )
}
