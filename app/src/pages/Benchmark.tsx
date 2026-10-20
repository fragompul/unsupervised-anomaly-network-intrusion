import { motion } from 'framer-motion'
import { useMemo, useState } from 'react'
import { CDDiagram } from '../components/charts/CDDiagram'
import { Heatmap } from '../components/charts/Heatmap'
import { Scatter } from '../components/charts/Scatter'
import { Badge, Card, Empty, PageHeader, SectionTitle, Segmented, Skeleton } from '../components/ui'
import { useData } from '../lib/context'
import { detTitle, familyOf, FAMILY_LABEL } from '../lib/detectors'
import { DATASET_ORDER, DATASET_SHORT, fmt } from '../lib/format'
import { CATEGORY_LABEL, FAMILY_COLOR, rampCss } from '../lib/palette'
import type { MainRow } from '../lib/types'

type Metric = 'roc' | 'pr' | 'tpr_at_1pct_fpr' | 'a5_f1'
const METRICS: { value: Metric; label: string; hint: string }[] = [
  { value: 'roc', label: 'ROC-AUC', hint: 'Threshold-free ranking quality' },
  { value: 'pr', label: 'PR-AUC', hint: 'Precision–recall, sensitive to prevalence' },
  { value: 'tpr_at_1pct_fpr', label: 'TPR @ 1% FPR', hint: 'Recall at the strictest alarm budget' },
  { value: 'a5_f1', label: 'F1 @ α=5%', hint: 'Conformal threshold, fixed before seeing test data' },
]

export default function Benchmark() {
  const { summary, manifest } = useData()
  const main = summary?.main
  const datasets = useMemo(() => DATASET_ORDER.filter((d) => main?.table.some((r) => r.dataset === d)), [main])
  const [dataset, setDataset] = useState<string | null>(null)
  const [regime, setRegime] = useState<'clean' | 'natural'>('clean')
  const [metric, setMetric] = useState<Metric>('roc')
  const [showEns, setShowEns] = useState(true)
  const ds = dataset ?? datasets[0] ?? ''

  const rows = useMemo(() => {
    if (!main) return []
    return main.table.filter((r) => r.dataset === ds && r.regime === regime && (showEns || !r.detector.startsWith('ens_'))).sort((a, b) => b[metric] - a[metric])
  }, [main, ds, regime, metric, showEns])

  const titles = summary?.titles ?? {}
  const T = (id: string) => detTitle(id, titles)
  const best = rows[0]?.[metric] ?? 1
  const worst = Math.min(...rows.map((r) => r[metric]), 0.5)
  const scale = (v: number) => Math.max(0.02, (v - (metric === 'roc' || metric === 'pr' ? Math.max(0.4, worst - 0.05) : 0)) / (best - (metric === 'roc' || metric === 'pr' ? Math.max(0.4, worst - 0.05) : 0)))

  const cats = useMemo(() => {
    if (!main) return { cols: [] as string[], rowsIds: [] as string[], values: [] as (number | null)[][] }
    const sel = main.categories.filter((c) => c.dataset === ds && c.regime === regime)
    const cols = Array.from(new Set(sel.map((c) => c.category))).sort((a, b) => (a === 'normal' ? 1 : b === 'normal' ? -1 : a.localeCompare(b)))
    const ids = rows.map((r) => r.detector)
    const values = ids.map((id) => cols.map((c) => sel.find((s) => s.detector === id && s.category === c)?.recall_a5 ?? null))
    return { cols, rowsIds: ids, values }
  }, [main, ds, regime, rows])

  if (!summary || !main) return <Shell><Skeleton className="h-[500px]" /></Shell>
  if (!datasets.length) return <Shell><Empty>Benchmark results are not available yet.</Empty></Shell>

  const stats = main.stats.all_seeds
  const nSeeds = rows[0]?.n_seeds ?? 0
  const familyOfId = (id: string) => familyOf(id)

  const dots = rows.filter((r) => r.score_ms_per_1k != null && !r.detector.startsWith('ens_')).map((r) => ({
    id: r.detector, label: T(r.detector), x: Math.max(0.02, r.score_ms_per_1k as number), y: r.roc, color: FAMILY_COLOR[familyOfId(r.detector)] ?? '#8f98b3', r: 6,
  }))

  return (
    <Shell>
      <div className="mb-8 flex flex-wrap items-center gap-3">
        <Segmented label="dataset" value={ds} onChange={setDataset} options={datasets.map((d) => ({ value: d, label: DATASET_SHORT[d] ?? d }))} />
        <Segmented label="regime" value={regime} onChange={setRegime} options={[{ value: 'clean', label: 'Clean training' }, { value: 'natural', label: 'Natural (contaminated)' }]} />
        <Segmented label="metric" value={metric} onChange={setMetric} options={METRICS.map((m) => ({ value: m.value, label: m.label }))} />
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-[13px] text-muted">
          <input type="checkbox" checked={showEns} onChange={(e) => setShowEns(e.target.checked)} className="accent-violet" /> Show conformal ensembles
        </label>
      </div>

      <Card className="mb-6 p-6">
        <SectionTitle hint={`${nSeeds} seeds · ${METRICS.find((m) => m.value === metric)?.hint}`}>Leaderboard · {DATASET_SHORT[ds]} · {regime === 'clean' ? 'normal-only training' : 'training split as released'}</SectionTitle>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[760px] border-separate border-spacing-y-1 text-[13px]">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wider text-faint">
                <th className="w-8 pb-1 font-medium">#</th>
                <th className="pb-1 font-medium">Detector</th>
                <th className="w-[38%] pb-1 font-medium">{METRICS.find((m) => m.value === metric)?.label} <span className="normal-case tracking-normal">(mean ± 95% CI over seeds)</span></th>
                <th className="pb-1 text-right font-medium" title="Realized false-positive rate when flagging at a conformal α of 1%">FPR @ <span className="normal-case">α</span>=1%</th>
                <th className="pb-1 text-right font-medium">TPR @ <span className="normal-case">α</span>=5%</th>
                <th className="pb-1 text-right font-medium">Fit</th>
                <th className="pb-1 text-right font-medium">ms / 1k</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <LeaderRow key={r.detector} r={r} i={i} metric={metric} scale={scale} title={T(r.detector)} family={familyOfId(r.detector)} />
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-faint">
          FPR @ α=1% is the <em>honesty check</em>: a conformal detector calibrated on clean normal flows promises ≤ 1% false alarms on exchangeable traffic. Red means the promise broke on this test split — that is distribution shift, not noise.
        </p>
      </Card>

      <div className="mb-6 grid gap-6 xl:grid-cols-[1.35fr_1fr]">
        <Card className="p-6">
          <SectionTitle hint={stats ? `Friedman χ² = ${stats.statistic.toFixed(1)}, p ${stats.p < 1e-4 ? '< 10⁻⁴' : '= ' + stats.p.toFixed(4)} · ${stats.n_blocks} blocks (dataset × regime × seed)` : undefined}>Critical-difference diagram · ROC-AUC</SectionTitle>
          {stats ? (
            <>
              <CDDiagram ranks={Object.fromEntries(Object.entries(stats.mean_ranks).filter(([id]) => showEns || !id.startsWith('ens_')))} cd={stats.cd} groups={stats.cd_groups} title={T} family={familyOfId} />
              <p className="mt-2 text-xs leading-relaxed text-faint">Lower rank is better. Methods joined by a violet bar are <em>not</em> significantly different (Nemenyi, α = 0.05). Seeds are treated as replicates, which is optimistic — see the paper for the dataset-level analysis.</p>
            </>
          ) : <Empty>Not enough blocks for a Friedman test yet.</Empty>}
        </Card>
        <Card className="p-6">
          <SectionTitle hint="Holm-corrected Wilcoxon vs. the best-ranked detector">Who is statistically behind?</SectionTitle>
          <div className="mt-2 max-h-[360px] space-y-1 overflow-y-auto pr-1">
            {(main.stats.holm_vs_best ?? []).slice().sort((a, b) => a.p_holm - b.p_holm).map((h) => (
              <div key={h.method} className="flex items-center gap-3 rounded-lg px-2 py-1.5 text-[13px] hover:bg-white/[0.04]">
                <span className="h-2 w-2 rounded-full" style={{ background: FAMILY_COLOR[familyOfId(h.method)] }} />
                <span className="flex-1 truncate">{T(h.method)}</span>
                <span className="num text-xs text-muted">Δ {fmt.signed(h.mean_diff)}</span>
                <Badge color={h.significant ? '#fb7185' : '#8f98b3'}>{h.significant ? 'worse' : 'n.s.'}</Badge>
              </div>
            ))}
          </div>
          <p className="mt-2 text-xs text-faint">Reference: <span className="text-text">{stats ? T(main.stats.best_by_rank ?? '') : '—'}</span>.</p>
        </Card>
      </div>

      <div className="mb-6 grid gap-6 xl:grid-cols-[1.35fr_1fr]">
        <Card className="p-6">
          <SectionTitle hint="Share of each attack family flagged at α = 5% (calibrated on clean normals)">Where each detector goes blind</SectionTitle>
          {cats.cols.length ? (
            <Heatmap rows={cats.rowsIds} cols={cats.cols} values={cats.values} rowLabel={T} colLabel={(c) => CATEGORY_LABEL(c)} color={(v) => rampCss(Math.min(1, v * 0.95))} format={(v) => `${Math.round(v * 100)}`} labelW={150} cellH={26} textColor={(v) => (v > 0.7 ? '#10131f' : 'rgba(255,255,255,0.9)')} />
          ) : <Empty>No category data.</Empty>}
          <p className="mt-2 text-xs text-faint">Cells show recall %. The <em>normal</em> column is the false-positive rate — it should stay near 5%.</p>
        </Card>
        <Card className="p-6">
          <SectionTitle hint="Scoring latency per 1,000 flows (log scale, CPU)">Accuracy vs. cost</SectionTitle>
          {dots.length ? <Scatter dots={dots} xLabel="ms per 1,000 flows (log)" yLabel="ROC-AUC" xLog fx={(v) => (v < 1 ? v.toFixed(2) : v.toFixed(0))} xTicks={[0.1, 1, 10, 100, 1000]} /> : <Empty>No timing data.</Empty>}
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
            {Object.entries(FAMILY_LABEL).filter(([k]) => k !== 'ensemble' && k !== 'robust-wrapper').map(([k, v]) => (
              <span key={k} className="flex items-center gap-1.5 text-[11px] text-muted"><span className="h-2 w-2 rounded-full" style={{ background: FAMILY_COLOR[k] }} />{v}</span>
            ))}
          </div>
        </Card>
      </div>
      <p className="text-xs text-faint">{manifest ? `Generated ${new Date(manifest.generated).toUTCString().slice(0, 16)} · Noctua v${manifest.version}` : ''}</p>
    </Shell>
  )
}

function LeaderRow({ r, i, metric, scale, title, family }: { r: MainRow; i: number; metric: Metric; scale: (v: number) => number; title: string; family: string }) {
  const color = FAMILY_COLOR[family] ?? '#8f98b3'
  const v = r[metric]
  const isEns = family === 'ensemble'
  const ci = metric === 'roc' ? r.roc_ci : undefined
  const broken = r.a1_fpr > 0.02
  return (
    <motion.tr initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: Math.min(i * 0.02, 0.4) }} className="group">
      <td className="rounded-l-lg bg-white/[0.02] py-2 pl-3 num text-faint group-hover:bg-white/[0.05]">{i + 1}</td>
      <td className="bg-white/[0.02] py-2 group-hover:bg-white/[0.05]">
        <div className="flex items-center gap-2.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: color, boxShadow: `0 0 10px ${color}88` }} />
          <span className={isEns ? 'font-semibold text-white' : 'font-medium'}>{title}</span>
          <span className="hidden text-[11px] text-faint xl:inline">{FAMILY_LABEL[family]}</span>
        </div>
      </td>
      <td className="bg-white/[0.02] py-2 pr-4 group-hover:bg-white/[0.05]">
        <div className="flex items-center gap-3">
          <div className="relative h-2 flex-1 rounded-full bg-white/[0.06]">
            <motion.div initial={{ width: 0 }} animate={{ width: `${scale(v) * 100}%` }} transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }} className="h-full rounded-full" style={{ background: `linear-gradient(90deg, ${color}55, ${color})` }} />
            {ci && <div className="absolute top-1/2 h-3 -translate-y-1/2 border-x-2 border-white/70" style={{ left: `${scale(ci[0]) * 100}%`, width: `${Math.max(0.5, (scale(ci[1]) - scale(ci[0])) * 100)}%` }}><div className="absolute top-1/2 h-px w-full bg-white/70" /></div>}
          </div>
          <span className="num w-[104px] whitespace-nowrap text-right text-[13px]">{fmt.auc(v)}{r.roc_std > 0 && metric === 'roc' ? <span className="text-faint"> ±{r.roc_std.toFixed(3)}</span> : null}</span>
        </div>
      </td>
      <td className="bg-white/[0.02] py-2 text-right num group-hover:bg-white/[0.05]" style={{ color: broken ? '#fb7185' : '#8f98b3' }}>{fmt.pct(r.a1_fpr, 2)}</td>
      <td className="bg-white/[0.02] py-2 text-right num text-muted group-hover:bg-white/[0.05]">{fmt.pct(r.a5_tpr, 1)}</td>
      <td className="bg-white/[0.02] py-2 text-right num text-muted group-hover:bg-white/[0.05]">{fmt.secs(r.fit_s)}</td>
      <td className="rounded-r-lg bg-white/[0.02] py-2 pr-3 text-right num text-muted group-hover:bg-white/[0.05]">{r.score_ms_per_1k != null ? (r.score_ms_per_1k < 1 ? r.score_ms_per_1k.toFixed(2) : r.score_ms_per_1k.toFixed(1)) : '—'}</td>
    </motion.tr>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-[1500px] px-6 pb-24 pt-28">
      <PageHeader eyebrow="Benchmark" title={<>Fifteen detectors. <span className="grad-text">No single-number leaderboard.</span></>}>
        Every cell is the mean over independent seeds with held-out calibration, bootstrap intervals and rank-based significance tests. Switch training regime to see how much each method depends on a curated clean window.
      </PageHeader>
      {children}
    </div>
  )
}
