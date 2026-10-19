import { useEffect, useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Heatmap } from '../components/charts/Heatmap'
import { LineChart, type Series } from '../components/charts/LineChart'
import { Scatter } from '../components/charts/Scatter'
import { Badge, Card, Empty, PageHeader, SectionTitle, Segmented, Select, Skeleton, cx } from '../components/ui'
import { useData } from '../lib/context'
import { detTitle } from '../lib/detectors'
import { DATASET_SHORT, fmt } from '../lib/format'
import { detColor, rampCss } from '../lib/palette'

const SUBSET = ['ae', 'vae', 'knn', 'iforest', 'gmm', 'pca', 'ecod', 'deep_svdd', 'flow', 'neutral']

export default function Robustness() {
  const { summary } = useData()
  const loc = useLocation()
  useEffect(() => {
    if (!loc.hash) return
    const id = loc.hash.replace('#', '')
    const t = setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 350)
    return () => clearTimeout(t)
  }, [loc.hash, summary])

  return (
    <div className="mx-auto max-w-[1500px] px-6 pb-24 pt-28">
      <PageHeader eyebrow="Robustness lab" title={<>Three ways the textbook setting <span className="grad-text">stops being true.</span></>}>
        Unsupervised detectors assume a clean training window, stationary traffic and the same network at deployment. Each experiment below removes one of those assumptions and measures what survives — including the conformal false-positive guarantee.
      </PageHeader>
      {!summary ? <Skeleton className="h-96" /> : (
        <div className="space-y-16">
          <Contamination />
          <Drift />
          <Transfer />
          <Calibration />
        </div>
      )}
    </div>
  )
}

function SectionHead({ id, n, title, text }: { id: string; n: string; title: string; text: string }) {
  return (
    <div id={id} className="mb-6 max-w-3xl scroll-mt-24">
      <div className="mb-2 flex items-center gap-3"><span className="num flex h-7 w-7 items-center justify-center rounded-lg bg-violet/15 text-xs text-violet-soft">{n}</span><h2 className="text-2xl font-semibold tracking-tight">{title}</h2></div>
      <p className="text-[14.5px] leading-relaxed text-muted">{text}</p>
    </div>
  )
}

// -------------------------------------------------------------------------------- contamination

function Contamination() {
  const { summary } = useData()
  const c = summary?.contamination
  const datasets = useMemo(() => Array.from(new Set(c?.table.map((r) => r.dataset) ?? [])), [c])
  const [ds, setDs] = useState('')
  const [showCtr, setShowCtr] = useState(true)
  const [active, setActive] = useState<string[]>(['ae', 'knn', 'iforest', 'flow'])
  const dataset = ds || datasets[0] || ''
  if (!c || !datasets.length) return <section><SectionHead id="contamination" n="1" title="Hidden attacks in the training window" text="Attacks are injected into the normal training base at an exact rate ρ — the model never sees a label." /><Empty>Contamination results are not available yet.</Empty></section>

  const rhos = Array.from(new Set(c.table.filter((r) => r.dataset === dataset).map((r) => r.rho))).sort((a, b) => a - b)
  const rowsOf = (id: string) => c.table.filter((r) => r.dataset === dataset && r.detector === id).sort((a, b) => a.rho - b.rho)
  const series: Series[] = []
  for (const id of active) {
    const base = rowsOf(id)
    if (base.length) series.push({ id, label: detTitle(id, summary?.titles), color: detColor(id), points: base.map((r) => [r.rho * 100, r.roc]), band: base.map((r) => [r.rho * 100, r.roc - r.roc_std, r.roc + r.roc_std]) })
    if (showCtr) {
      const ctr = rowsOf(`${id}+ctr0.1`)
      if (ctr.length) series.push({ id: `${id}+ctr0.1`, label: `${detTitle(id, summary?.titles)} + CTR`, color: detColor(id), dashed: true, width: 2.4, points: ctr.map((r) => [r.rho * 100, r.roc]) })
    }
  }
  const gains = c.ctr_gain.filter((g) => g.dataset === dataset && Math.abs(g.rho - 0.05) < 1e-9 && g.detector.endsWith('+ctr0.1')).sort((a, b) => b.gain - a.gain)
  const abl = ['ae', 'ae+ctr0.05', 'ae+ctr0.1', 'ae+ctr0.2', 'ae+ctr0.1[iforest]', 'ae+ctr0.1[ecod]', 'ae+ctr0.1[pca]', 'ae+ctr@rho']
  const ablRows = abl.map((id) => ({ id, row: c.table.find((r) => r.dataset === dataset && r.detector === id && Math.abs(r.rho - 0.05) < 1e-9) })).filter((r) => r.row)
  const ablLabel: Record<string, string> = { ae: 'AE (no purification)', 'ae+ctr0.05': 'CTR · trim 5%', 'ae+ctr0.1': 'CTR · trim 10% (default)', 'ae+ctr0.2': 'CTR · trim 20%', 'ae+ctr0.1[iforest]': 'screener: Isolation Forest only', 'ae+ctr0.1[ecod]': 'screener: ECOD only', 'ae+ctr0.1[pca]': 'screener: PCA only', 'ae+ctr@rho': 'oracle trim = true ρ (headroom)' }

  return (
    <section>
      <SectionHead id="contamination" n="1" title="Hidden attacks in the training window" text="Attacks are injected into the normal training base at an exact rate ρ — the model never sees a label. Reconstruction and density methods learn the injected attack as ‘normal’, and the damage is steep. Consensus-Trimmed Refit (CTR) screens the training set with three independent detectors before fitting." />
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <Segmented label="cds" value={dataset} onChange={setDs} options={datasets.map((d) => ({ value: d, label: DATASET_SHORT[d] ?? d }))} />
        <label className="flex cursor-pointer items-center gap-2 text-[13px] text-muted"><input type="checkbox" className="accent-violet" checked={showCtr} onChange={(e) => setShowCtr(e.target.checked)} /> Overlay CTR (dashed)</label>
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">
        <Card className="p-6">
          <div className="mb-3 flex flex-wrap gap-1.5">
            {SUBSET.map((id) => {
              const on = active.includes(id)
              return (
                <button key={id} onClick={() => setActive(on ? active.filter((x) => x !== id) : [...active, id])} className={cx('focus-ring flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-all', on ? 'border-white/25 bg-white/[0.08] text-text' : 'border-line text-faint hover:text-muted')}>
                  <span className="h-2 w-2 rounded-full" style={{ background: detColor(id), opacity: on ? 1 : 0.4 }} />{detTitle(id, summary?.titles)}
                </button>
              )
            })}
          </div>
          <LineChart height={340} series={series} xDomain={[0, Math.max(...rhos) * 100]} yDomain={[0.4, 1]} xLabel="Contamination ρ of the training window (%)" yLabel="ROC-AUC" formatX={(v) => `${v}%`} formatY={(v) => v.toFixed(2)} xTicks={rhos.map((r) => r * 100)} hoverX markers refLines={[{ y: 0.5, label: 'chance' }]} />
          <p className="mt-2 text-xs text-faint">Lines: mean over seeds (band: ±1 std). Dashed: same detector behind CTR.</p>
        </Card>
        <div className="space-y-6">
          <Card className="p-6">
            <SectionTitle hint="ρ = 5%, mean over seeds">CTR gain over the plain detector</SectionTitle>
            <div className="mt-3 space-y-2.5">
              {gains.map((g) => (
                <div key={g.detector} className="grid grid-cols-[110px_1fr_64px] items-center gap-3 text-[13px]">
                  <span className="truncate">{detTitle(g.base, summary?.titles)}</span>
                  <div className="relative h-2 rounded-full bg-white/[0.06]">
                    <div className="absolute top-0 h-full rounded-full" style={{ left: g.gain >= 0 ? '30%' : `${30 + g.gain * 150}%`, width: `${Math.min(70, Math.abs(g.gain) * 150)}%`, background: g.gain >= 0 ? '#a3e635' : '#fb7185' }} />
                    <div className="absolute left-[30%] top-[-3px] h-[14px] w-px bg-white/30" />
                  </div>
                  <span className="num text-right text-xs" style={{ color: g.gain >= 0 ? '#a3e635' : '#fb7185' }}>{fmt.signed(g.gain)}</span>
                </div>
              ))}
              {!gains.length && <Empty>No CTR rows.</Empty>}
            </div>
          </Card>
          <Card className="p-6">
            <SectionTitle hint="Autoencoder, ρ = 5%">Ablation: why a committee?</SectionTitle>
            <div className="mt-3 space-y-2">
              {ablRows.map(({ id, row }) => (
                <div key={id} className="grid grid-cols-[1fr_54px] items-center gap-3 text-[12.5px]">
                  <div>
                    <div className={cx('mb-1 truncate', id === 'ae+ctr0.1' ? 'font-medium text-text' : 'text-muted')}>{ablLabel[id]}</div>
                    <div className="h-1.5 rounded-full bg-white/[0.06]"><div className="h-full rounded-full" style={{ width: `${Math.max(2, ((row!.roc - 0.4) / 0.6) * 100)}%`, background: id === 'ae+ctr0.1' ? 'linear-gradient(90deg,#8b7bff,#22d3ee)' : id === 'ae+ctr@rho' ? '#a3e635' : 'rgba(255,255,255,0.35)' }} /></div>
                  </div>
                  <span className="num text-right">{fmt.auc(row!.roc)}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </section>
  )
}

// -------------------------------------------------------------------------------------- drift

function Drift() {
  const { summary } = useData()
  const d = summary?.drift
  const days = ['Tue', 'Wed', 'Thu', 'Fri']
  if (!d) return <section><SectionHead id="drift" n="2" title="A week of drift" text="Fit on Monday’s attack-free traffic, calibrate on Monday, then watch Tuesday to Friday." /><Empty>Drift results are not available yet.</Empty></section>

  const dets = Array.from(new Set(d.table.map((r) => r.detector))).filter((x) => !x.startsWith('ens_'))
  const ordered = [...dets].sort((a, b) => {
    const m = (id: string) => d.table.filter((r) => r.detector === id).reduce((s, r) => s + r.fpr_a1, 0)
    return m(a) - m(b)
  })
  const get = (id: string, day: string, k: 'fpr_a1' | 'psi_benign' | 'roc') => d.table.find((r) => r.detector === id && r.day === day)?.[k] ?? null
  const fprVals = ordered.map((id) => days.map((day) => get(id, day, 'fpr_a1')))
  const dots = d.table.filter((r) => !r.detector.startsWith('ens_') && r.fpr_a1 > 0).map((r) => ({ id: `${r.detector}-${r.day}`, label: '', x: Math.max(0.001, r.psi_benign), y: Math.log10(Math.max(1e-4, r.fpr_a1)), color: detColor(r.detector), r: 4 }))
  const alarm = d.drift_alarm

  return (
    <section>
      <SectionHead id="drift" n="2" title="A week of drift" text="Fit on Monday’s attack-free CIC-IDS2017 traffic and calibrate a conformal threshold on held-out Monday flows (nominal false-positive rate 1%). Then replay Tuesday to Friday: the benign traffic itself changes, and the guarantee quietly stops holding." />
      <div className="grid gap-6 xl:grid-cols-[1.3fr_1fr]">
        <Card className="p-6">
          <SectionTitle hint="Realized false-positive rate on benign flows · nominal 1%">Calibration decay by day</SectionTitle>
          <Heatmap rows={ordered} cols={days} values={fprVals} rowLabel={(id) => detTitle(id, summary?.titles)} color={(v) => rampCss(Math.min(1, Math.log10(Math.max(v, 0.002) / 0.002) / 2.2))} format={(v) => `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`} cellH={26} labelW={150} textColor={(v) => (v > 0.1 ? '#10131f' : 'rgba(255,255,255,0.92)')} />
          <p className="mt-2 text-xs text-faint">A cell near 1% means the guarantee held. Values above ~3% are a broken promise, and every detector breaks somewhere.</p>
        </Card>
        <Card className="p-6">
          <SectionTitle hint={d.psi_fpr_spearman ? `Spearman ρ = ${d.psi_fpr_spearman.rho.toFixed(2)}` : undefined}>A label-free alarm</SectionTitle>
          <p className="mb-2 text-xs leading-relaxed text-muted">Population Stability Index of benign scores vs. Monday needs <em>no labels</em>, yet tracks the false-positive blow-up.</p>
          <Scatter dots={dots} xLabel="PSI of benign scores (log)" yLabel="log₁₀ realized FPR" xLog height={250} fx={(v) => (v < 0.1 ? v.toFixed(2) : v.toFixed(1))} fy={(v) => v.toFixed(1)} xTicks={[0.01, 0.1, 1, 10]} />
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge color="#f5b84b">PSI ≥ {alarm.threshold_psi} alarm</Badge>
            {alarm.recall != null && <Badge color="#34d399">recall {fmt.pct(alarm.recall, 0)}</Badge>}
            {alarm.precision != null && <Badge color="#60a5fa">precision {fmt.pct(alarm.precision, 0)}</Badge>}
          </div>
        </Card>
      </div>
    </section>
  )
}

// ----------------------------------------------------------------------------------- transfer

function Transfer() {
  const { summary } = useData()
  const t = summary?.transfer
  if (!t) return <section><SectionHead id="transfer" n="3" title="A network it has never seen" text="Train on one dataset’s clean normals and detect on another." /><Empty>Transfer results are not available yet.</Empty></section>
  type Row = NonNullable<typeof t>['table'][number]
  const configs: { key: string; label: string; match: (r: Row) => boolean }[] = [
    { key: 'u-u', label: 'UNSW → UNSW', match: (r) => r.source === 'unsw_nb15' && r.target === 'unsw_nb15' },
    { key: 'u-c-s', label: 'UNSW → CIC', match: (r) => r.source === 'unsw_nb15' && r.target === 'cicids2017' && r.scaler === 'source' },
    { key: 'u-c-t', label: 'UNSW → CIC · domain-std', match: (r) => r.source === 'unsw_nb15' && r.target === 'cicids2017' && r.scaler === 'target' },
    { key: 'c-c', label: 'CIC → CIC', match: (r) => r.source === 'cicids2017' && r.target === 'cicids2017' },
    { key: 'c-u-s', label: 'CIC → UNSW', match: (r) => r.source === 'cicids2017' && r.target === 'unsw_nb15' && r.scaler === 'source' },
    { key: 'c-u-t', label: 'CIC → UNSW · domain-std', match: (r) => r.source === 'cicids2017' && r.target === 'unsw_nb15' && r.scaler === 'target' },
  ]
  const dets = Array.from(new Set(t.table.map((r) => r.detector))).filter((id) => !id.startsWith('ens_'))
  const val = (id: string, c: (typeof configs)[number]) => t.table.find((r) => r.detector === id && c.match(r))?.roc ?? null
  const ordered = [...dets].sort((a, b) => (val(b, configs[1]) ?? 0) + (val(b, configs[4]) ?? 0) - ((val(a, configs[1]) ?? 0) + (val(a, configs[4]) ?? 0)))
  const values = ordered.map((id) => configs.map((c) => val(id, c)))
  const meanOf = (c: (typeof configs)[number]) => { const v = dets.map((id) => val(id, c)).filter((x): x is number => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null }
  return (
    <section>
      <SectionHead id="transfer" n="3" title="A network it has never seen" text="UNSW-NB15 and CIC-IDS2017 are mapped onto 18 physically aligned flow features. A detector fitted on one capture’s clean normals is pointed at the other. Standardizing each domain with its own unlabeled statistics (domain-std) recovers part of the gap." />
      <Card className="p-6">
        <SectionTitle hint="ROC-AUC, mean over seeds · detectors sorted by transfer performance">In-dataset (diagonal blocks) vs. cross-dataset</SectionTitle>
        <Heatmap rows={ordered} cols={configs.map((c) => c.key)} colLabel={(k) => configs.find((c) => c.key === k)!.label} values={values} rowLabel={(id) => detTitle(id, summary?.titles)} color={(v) => rampCss(Math.max(0, (v - 0.45) / 0.55))} format={(v) => v.toFixed(2)} cellH={26} labelW={150} textColor={(v) => (v > 0.7 ? '#10131f' : 'rgba(255,255,255,0.92)')} />
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-6">
          {configs.map((c) => (
            <div key={c.key}><div className="text-[10px] uppercase tracking-wider text-faint">{c.label}</div><div className="num text-lg">{fmt.auc(meanOf(c))}</div><div className="text-[10px] text-faint">mean across detectors</div></div>
          ))}
        </div>
      </Card>
    </section>
  )
}

// ------------------------------------------------------------------------------- calibration

function Calibration() {
  const { summary } = useData()
  const main = summary?.main
  const [det, setDet] = useState('ae')
  const datasets = Array.from(new Set(main?.coverage.map((c) => c.dataset) ?? []))
  const [ds, setDs] = useState('')
  const dataset = ds || datasets[0] || ''
  if (!main) return null
  const inDs = main.coverage.filter((c) => c.dataset === dataset && c.regime === 'clean' && c.detector === det).sort((a, b) => a.alpha - b.alpha)
  const driftRows = (day: string) => (summary?.drift?.coverage ?? []).filter((c) => c.detector === det && c.day === day).sort((a, b) => a.alpha - b.alpha)
  const series: Series[] = []
  if (inDs.length) series.push({ id: 'in', label: `In-dataset · ${DATASET_SHORT[dataset] ?? dataset}`, color: '#34d399', points: inDs.map((c) => [c.alpha, c.fpr]) })
  if (driftRows('Tue').length) series.push({ id: 'tue', label: 'CIC · Tuesday (1 day later)', color: '#f5b84b', points: driftRows('Tue').map((c) => [c.alpha, c.fpr]) })
  if (driftRows('Fri').length) series.push({ id: 'fri', label: 'CIC · Friday (4 days later)', color: '#fb7185', points: driftRows('Fri').map((c) => [c.alpha, c.fpr]) })
  const options = Array.from(new Set(main.coverage.map((c) => c.detector))).map((id) => ({ value: id, label: detTitle(id, summary?.titles) }))
  return (
    <section>
      <SectionHead id="calibration" n="4" title="Does the conformal guarantee hold?" text="A split-conformal detector promises that at nominal level α, at most a fraction α of normal flows are flagged — for any detector, with no distributional assumptions, as long as calibration and test traffic are exchangeable. The reliability diagram shows exactly where that assumption fails: on or below the diagonal is a kept promise." />
      <Card className="p-6">
        <div className="mb-4 flex flex-wrap items-end gap-4">
          <Select className="w-56" label="Detector" value={det} onChange={setDet} options={options} />
          {datasets.length > 1 && <Segmented label="cal-ds" value={dataset} onChange={setDs} options={datasets.map((d) => ({ value: d, label: DATASET_SHORT[d] ?? d }))} />}
        </div>
        <LineChart height={360} series={series} xDomain={[0, 0.2]} yDomain={[0, 0.5]} xLabel="Nominal α (allowed false-positive rate)" yLabel="Realized FPR on normal flows" formatX={(v) => `${(v * 100).toFixed(0)}%`} formatY={(v) => `${(v * 100).toFixed(0)}%`} diagonal hoverX />
        <p className="mt-2 text-xs text-faint">Dashed diagonal = a perfectly valid detector. Curves above it are guarantee violations caused by shift between calibration and test.</p>
      </Card>
    </section>
  )
}
