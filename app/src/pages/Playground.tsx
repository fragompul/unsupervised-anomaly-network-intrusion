import { motion } from 'framer-motion'
import { Dices, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Badge, Card, PageHeader, Segmented, Skeleton } from '../components/ui'
import { scoreFlow } from '../lib/ae'
import { useData } from '../lib/context'
import { featureZ, nlogp } from '../lib/data'
import { DATASET_SHORT, fmt } from '../lib/format'
import { useDataset } from '../lib/hooks'
import { CATEGORY_LABEL, categoryColor } from '../lib/palette'
import type { DatasetData } from '../lib/types'

const KEY_FEATURES = 14
const clean = (n: string) => n.replace(/^(num|cat)__/, '').replace(/_/g, ' ')

function flowVector(ds: DatasetData, i: number): number[] {
  return Array.from({ length: ds.meta.n_features }, (_, j) => featureZ(ds, i, j))
}

export default function Playground() {
  const { manifest } = useData()
  const [params, setParams] = useSearchParams()
  const [key, setKey] = useState(params.get('dataset') ?? '')
  useEffect(() => {
    if (manifest && !manifest.datasets.some((d) => d.key === key)) setKey(manifest.datasets[0]?.key ?? '')
  }, [manifest, key])
  const { data: ds } = useDataset(manifest, key)
  const [x, setX] = useState<number[] | null>(null)
  const [origin, setOrigin] = useState<string>('')

  const keyFeatures = useMemo(() => {
    if (!ds) return []
    const counts = new Float64Array(ds.meta.n_features)
    for (let i = 0; i < ds.n; i++) for (let r = 0; r < 5; r++) counts[ds.attribution[i * 10 + r]] += ds.attribution[i * 10 + 5 + r] / 255
    // sliders only make sense for continuous features; one-hot columns stay visible in the attribution
    return Array.from(counts, (c, j) => [c, j] as const).filter(([, j]) => !ds.meta.feature_names[j].startsWith('cat__')).sort((a, b) => b[0] - a[0]).slice(0, KEY_FEATURES).map(([, j]) => j)
  }, [ds])

  // presets: per category, the sampled flow whose AE evidence is closest to the category median
  const presets = useMemo(() => {
    if (!ds) return []
    const ia = ds.meta.detectors.findIndex((d) => d.id === 'ae')
    return ds.meta.categories.map((c, ci) => {
      const idx: number[] = []
      for (let i = 0; i < ds.n && idx.length < 400; i++) if (ds.category[i] === ci) idx.push(i)
      const vals = idx.map((i) => nlogp(ds, Math.max(0, ia), i)).sort((a, b) => a - b)
      const med = vals[Math.floor(vals.length / 2)] ?? 0
      const best = idx.reduce((b, i) => (Math.abs(nlogp(ds, Math.max(0, ia), i) - med) < Math.abs(nlogp(ds, Math.max(0, ia), b) - med) ? i : b), idx[0])
      return { category: c, index: best, ci }
    }).filter((p) => p.index != null)
  }, [ds])

  useEffect(() => {
    if (!ds) return
    const flow = params.get('flow')
    if (flow != null && +flow < ds.n && params.get('dataset') === ds.meta.key) {
      setX(flowVector(ds, +flow))
      setOrigin(`flow #${flow} from the Explorer`)
    } else if (presets.length) {
      setX(flowVector(ds, presets[0].index))
      setOrigin(`typical ${CATEGORY_LABEL(presets[0].category)} flow`)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ds, presets.length > 0])

  const result = useMemo(() => (ds && x ? scoreFlow(ds.ae, x, ds.meta.feature_clip) : null), [ds, x])
  const recon = useMemo(() => (result ? Array.from(result.errors, (e, j) => ({ j, e })) : null), [result])
  if (!manifest || !ds || !x || !result || !recon) return <Shell><Skeleton className="h-[540px]" /></Shell>

  const p = result.pValue
  const verdict = p <= 0.01 ? { t: 'Anomalous', c: '#fb7185' } : p <= 0.05 ? { t: 'Suspicious', c: '#f5b84b' } : { t: 'Normal', c: '#34d399' }
  const evidence = Math.min(5, -Math.log10(p))
  const top = [...recon].sort((a, b) => b.e - a.e).slice(0, 7)
  const maxE = Math.max(1e-9, top[0].e)
  const set = (j: number, v: number) => setX((cur) => cur!.map((c, k) => (k === j ? v : c)))
  const load = (i: number, label: string) => { setX(flowVector(ds, i)); setOrigin(label); setParams({}) }
  const randomize = () => {
    const ks = [...keyFeatures].sort(() => Math.random() - 0.5).slice(0, 3)
    setX((cur) => cur!.map((c, k) => (ks.includes(k) ? (Math.random() < 0.5 ? -1 : 1) * (3 + Math.random() * 3) : c)))
    setOrigin('hand-perturbed flow')
  }

  return (
    <Shell>
      <div className="mb-5 flex flex-wrap items-center gap-3">
        <Segmented label="pg-ds" value={key} onChange={(v) => { setKey(v); setParams({}) }} options={manifest.datasets.map((d) => ({ value: d.key, label: DATASET_SHORT[d.key] ?? d.title }))} />
        <span className="text-xs text-faint">Start from</span>
        {presets.map((pr) => (
          <button key={pr.category} onClick={() => load(pr.index, `typical ${CATEGORY_LABEL(pr.category)} flow`)} className="focus-ring flex items-center gap-1.5 rounded-full border border-line px-3 py-1 text-xs text-muted transition-colors hover:bg-white/[0.06] hover:text-text">
            <span className="h-2 w-2 rounded-full" style={{ background: categoryColor(pr.category, pr.ci) }} />{CATEGORY_LABEL(pr.category)}
          </button>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
        <Card className="p-6">
          <div className="mb-1 flex items-center justify-between">
            <h2 className="text-[15px] font-semibold">Feature sliders <span className="font-normal text-faint">· standardized z-scores</span></h2>
            <div className="flex gap-2">
              <button onClick={randomize} className="focus-ring glass-flat flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted hover:text-text"><Dices size={13} /> Perturb</button>
              <button onClick={() => { const pr = presets[0]; if (pr) load(pr.index, `typical ${CATEGORY_LABEL(pr.category)} flow`) }} className="focus-ring glass-flat flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted hover:text-text"><RotateCcw size={13} /> Reset</button>
            </div>
          </div>
          <p className="mb-4 text-xs text-faint">Origin: {origin}. The hollow tick marks what the autoencoder <em>expects</em> given the rest of the flow; the gap is the error.</p>
          <div className="space-y-4">
            {keyFeatures.map((j) => {
              const v = x[j]
              const e = result.errors[j]
              const expected = result.recon[j]
              const pct = (z: number) => ((Math.max(-5, Math.min(5, z)) + 5) / 10) * 100
              return (
                <div key={j}>
                  <div className="mb-1 flex items-baseline justify-between text-xs">
                    <span className="truncate text-text">{clean(ds.meta.feature_names[j])}</span>
                    <span className="num text-muted">z = {v.toFixed(2)} · err {e.toFixed(2)}</span>
                  </div>
                  <div className="relative h-6">
                    <div className="absolute inset-x-0 top-[11px] h-[3px] rounded-full bg-white/[0.09]" />
                    <div className="absolute top-[8px] h-[8px] rounded-sm opacity-45" style={{ left: `${Math.min(pct(v), pct(expected))}%`, width: `${Math.abs(pct(v) - pct(expected))}%`, background: '#fb7185' }} title="gap between the value and what the autoencoder expects" />
                    <div className="absolute top-[5px] h-[14px] w-[14px] -translate-x-1/2 rounded-full border-2 border-white/70" style={{ left: `${pct(expected)}%` }} title="autoencoder expects this value" />
                    <input type="range" min={-5} max={5} step={0.05} value={Math.max(-5, Math.min(5, v))} onChange={(ev) => set(j, +ev.target.value)} className="slider absolute inset-0" style={{ ['--fill' as string]: '0%' }} aria-label={clean(ds.meta.feature_names[j])} />
                  </div>
                </div>
              )
            })}
          </div>
        </Card>

        <div className="space-y-6">
          <Card className="p-6">
            <div className="flex items-center justify-between"><h2 className="text-[15px] font-semibold">Verdict</h2><Badge color="#8b7bff">runs in your browser</Badge></div>
            <Gauge evidence={evidence} color={verdict.c} />
            <div className="mt-2 text-center">
              <motion.div key={verdict.t} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="text-2xl font-semibold" style={{ color: verdict.c }}>{verdict.t}</motion.div>
              <div className="num mt-1 text-sm text-muted">conformal p = {fmt.p(p)} · score {result.score.toFixed(3)}</div>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center text-[11px] text-faint">
              <div className="rounded-lg bg-white/[0.03] py-2">p ≤ 1%<div className="mt-0.5 text-rose">anomalous</div></div>
              <div className="rounded-lg bg-white/[0.03] py-2">p ≤ 5%<div className="mt-0.5 text-amber">suspicious</div></div>
              <div className="rounded-lg bg-white/[0.03] py-2">p &gt; 5%<div className="mt-0.5 text-mint">normal</div></div>
            </div>
          </Card>
          <Card className="p-6">
            <h2 className="mb-3 text-[15px] font-semibold">Why — top error features</h2>
            <div className="space-y-2.5">
              {top.map((t) => (
                <div key={t.j}>
                  <div className="mb-0.5 flex justify-between text-xs"><span className="truncate">{clean(ds.meta.feature_names[t.j])}</span><span className="num text-muted">{((t.e / result.errors.reduce((a, b) => a + b, 0)) * 100).toFixed(0)}%</span></div>
                  <div className="h-1.5 rounded-full bg-white/[0.06]"><motion.div className="h-full rounded-full bg-gradient-to-r from-violet to-cyan" animate={{ width: `${(t.e / maxE) * 100}%` }} transition={{ duration: 0.25 }} /></div>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-faint">Same exact attribution as the Explorer. Scored with the exported autoencoder weights in JavaScript — identical to the PyTorch model to 3 decimals.</p>
          </Card>
        </div>
      </div>
    </Shell>
  )
}

function Gauge({ evidence, color }: { evidence: number; color: string }) {
  const frac = Math.min(1, evidence / 5)
  const R = 78
  const arc = (f: number) => {
    const a = Math.PI * (1 - f)
    return `${100 + R * Math.cos(a)},${100 - R * Math.sin(a)}`
  }
  return (
    <svg viewBox="0 0 200 118" className="mx-auto mt-2 w-full max-w-[290px]">
      <path d={`M ${arc(0)} A ${R} ${R} 0 0 1 ${arc(1)}`} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={13} strokeLinecap="round" />
      <motion.path d={`M ${arc(0)} A ${R} ${R} 0 0 1 ${arc(1)}`} fill="none" stroke={color} strokeWidth={13} strokeLinecap="round" initial={false} animate={{ pathLength: Math.max(0.02, frac) }} transition={{ type: 'spring', stiffness: 120, damping: 20 }} style={{ filter: `drop-shadow(0 0 8px ${color}88)` }} />
      {[0.01, 0.05].map((a) => { const f = -Math.log10(a) / 5; const a1 = Math.PI * (1 - f); return <line key={a} x1={100 + (R - 12) * Math.cos(a1)} y1={100 - (R - 12) * Math.sin(a1)} x2={100 + (R + 12) * Math.cos(a1)} y2={100 - (R + 12) * Math.sin(a1)} stroke="rgba(255,255,255,0.5)" strokeWidth={1.5} /> })}
      <text x={100} y={92} textAnchor="middle" className="num fill-text text-[28px]">{evidence.toFixed(1)}</text>
      <text x={100} y={108} textAnchor="middle" className="fill-faint text-[9px] uppercase tracking-widest">−log₁₀ p</text>
    </svg>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-[1500px] px-6 pb-24 pt-28">
      <PageHeader eyebrow="Playground" title={<>Edit a flow. <span className="grad-text">Watch the verdict move.</span></>}>
        No server involved: the autoencoder’s weights are shipped with the page and evaluated in JavaScript. Start from a real flow of any attack family, then drag its most diagnostic features.
      </PageHeader>
      {children}
    </div>
  )
}
