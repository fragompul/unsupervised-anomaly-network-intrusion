import { motion } from 'framer-motion'
import { ArrowRight, Activity, BarChart3, Box, FlaskConical, GitCompare, Play, SlidersHorizontal } from 'lucide-react'
import { Link } from 'react-router-dom'
import { REPO } from '../App'
import { GithubIcon } from '../components/Logo'
import { PointCloud } from '../components/three/PointCloud'
import { Card, cx } from '../components/ui'
import { useData } from '../lib/context'
import { useCountUp, useDataset } from '../lib/hooks'
import { fmt } from '../lib/format'

function Counter({ value, suffix = '', decimals = 0 }: { value: number; suffix?: string; decimals?: number }) {
  const v = useCountUp(value)
  return <>{v.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}{suffix}</>
}

const FEATURES = [
  { to: '/explorer', icon: Box, title: '3D Explorer', text: 'Fly through 36k real flows in WebGL. Morph PCA into UMAP, flag by false-positive budget, and open any flow to see why it was flagged.', color: '#8b7bff' },
  { to: '/benchmark', icon: BarChart3, title: 'Benchmark', text: 'Fifteen detectors, five seeds, critical-difference diagrams and bootstrap intervals — no single-number leaderboard.', color: '#22d3ee' },
  { to: '/compare', icon: GitCompare, title: 'Compare', text: 'Put any two detectors head to head: ROC, precision–recall, and exactly which flows one catches and the other misses.', color: '#34d399' },
  { to: '/robustness', icon: FlaskConical, title: 'Robustness lab', text: 'Hidden attacks in training, a week of drift, a new network. See which guarantees survive — and which break.', color: '#f5b84b' },
  { to: '/replay', icon: Activity, title: 'Live replay', text: 'Replay four days of traffic as a stream. Watch calibration decay and a label-free drift alarm fire before the false alarms do.', color: '#fb7185' },
  { to: '/playground', icon: SlidersHorizontal, title: 'Playground', text: 'Drag features and score a synthetic flow with the autoencoder running entirely in your browser.', color: '#a3e635' },
]

export default function Home() {
  const { manifest, summary } = useData()
  const heroKey = manifest?.datasets.find((d) => d.key === 'unsw_nb15')?.key ?? manifest?.datasets[0]?.key ?? ''
  const { data: ds } = useDataset(manifest, heroKey)

  const totalFlows = manifest?.datasets.reduce((a, d) => a + d.n_test, 0) ?? 0
  const nDet = manifest?.datasets[0]?.detectors.filter((d) => d.family !== 'ensemble').length ?? 15

  const best = summary?.main?.stats.best_by_rank
  const contam = summary?.contamination?.degradation
  const meanDrop = contam && contam.length ? contam.reduce((a, r) => a + r.drop, 0) / contam.length : null
  const gains = summary?.contamination?.ctr_gain.filter((g) => Math.abs(g.rho - 0.05) < 1e-9)
  const meanGain = gains && gains.length ? gains.reduce((a, r) => a + r.gain, 0) / gains.length : null
  const drift = summary?.drift
  const ret = summary?.transfer?.retention.filter((r) => r.scaler === 'source' && r.retention != null)
  const meanRet = ret && ret.length ? ret.reduce((a, r) => a + (r.retention ?? 0), 0) / ret.length : null

  return (
    <div>
      {/* ---------------------------------------------------------------- hero */}
      <section className="relative isolate flex min-h-[100vh] items-center overflow-hidden pt-14">
        <div className="pointer-events-none absolute inset-y-0 right-[-8%] -z-10 w-[78%] max-lg:right-0 max-lg:w-full max-lg:opacity-50">
          {ds && <PointCloud className="h-full w-full" ds={ds} embedding={ds.embeddings.umap ? 'umap' : 'pca'} colorBy="category" detectorIndex={0} alpha={0.05} hiddenCategories={[]} selected={null} ambient autoRotate rotateSpeed={0.35} pointScale={1.25} />}
        </div>
        <div className="pointer-events-none absolute inset-0 -z-10 bg-[linear-gradient(90deg,#070a12_8%,rgba(7,10,18,0.55)_46%,transparent_75%)]" />
        <div className="mx-auto w-full max-w-[1500px] px-6">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }} className="max-w-[640px]">
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-white/[0.04] px-3.5 py-1.5 text-xs font-medium text-muted backdrop-blur">
              <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-[pulse-ring_2s_infinite] rounded-full bg-mint" /><span className="relative inline-flex h-2 w-2 rounded-full bg-mint" /></span>
              Research lab · unsupervised network intrusion detection
            </div>
            <h1 className="text-[2.9rem] font-semibold leading-[1.02] tracking-[-0.035em] sm:text-[4.4rem]">
              Anomaly detection,<br />
              <span className="grad-text">under pressure.</span>
            </h1>
            <p className="mt-6 max-w-[560px] text-[17px] leading-relaxed text-muted">
              Noctua stress-tests {nDet} unsupervised detectors on three generations of intrusion data — with attacks hidden in training, a week of drift and a network it has never seen — and wraps them in conformal guarantees you can check.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link to="/explorer" className="focus-ring group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet to-[#5b8def] px-5 py-3 text-sm font-semibold text-white shadow-glow transition-transform hover:-translate-y-0.5">
                Open the 3D explorer <ArrowRight size={16} className="transition-transform group-hover:translate-x-0.5" />
              </Link>
              <Link to="/benchmark" className="focus-ring inline-flex items-center gap-2 rounded-xl border border-line-strong bg-white/[0.04] px-5 py-3 text-sm font-semibold text-text backdrop-blur transition-colors hover:bg-white/[0.08]">
                <Play size={14} className="text-violet-soft" /> See the benchmark
              </Link>
              <a href={REPO} target="_blank" rel="noreferrer" className="focus-ring inline-flex items-center gap-2 px-3 py-3 text-sm font-medium text-muted hover:text-text"><GithubIcon /> Source</a>
            </div>
          </motion.div>

          <motion.dl initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.25, ease: [0.22, 1, 0.36, 1] }} className="mt-16 grid max-w-[720px] grid-cols-2 gap-6 sm:grid-cols-4">
            {[
              { k: 'Detectors', v: nDet },
              { k: 'Datasets', v: manifest?.datasets.length ?? 3 },
              { k: 'Test flows', v: totalFlows },
              { k: 'Experiments', v: 4 },
            ].map((s) => (
              <div key={s.k} className="border-l border-line pl-4">
                <dd className="num text-3xl font-medium"><Counter value={s.v} /></dd>
                <dt className="mt-1 text-xs uppercase tracking-wider text-faint">{s.k}</dt>
              </div>
            ))}
          </motion.dl>
        </div>
      </section>

      {/* ---------------------------------------------------------------- findings */}
      <section className="mx-auto max-w-[1500px] px-6 py-24">
        <div className="mb-10 max-w-2xl">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-violet-soft">Headline findings</div>
          <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">What breaks when the textbook assumptions do.</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Finding accent="#8b7bff" big={meanDrop != null ? `−${(meanDrop * 100).toFixed(0)} pts` : '—'} title="Hidden attacks poison the baseline" text="Average ROC-AUC lost when just 5% of the training window is silently malicious — reconstruction and density detectors learn the attack as normal." to="/robustness#contamination" />
          <Finding accent="#a3e635" big={meanGain != null ? `${meanGain >= 0 ? '+' : '−'}${Math.abs(meanGain * 100).toFixed(1)} pts` : '—'} title="Consensus trimming wins it back" text="Consensus-Trimmed Refit — a label-free screen by three independent detectors — recovers part of the loss at 5% contamination." to="/robustness#contamination" />
          <Finding accent="#fb7185" big={drift?.psi_fpr_spearman ? `ρ = ${drift.psi_fpr_spearman.rho.toFixed(2)}` : '—'} title="Drift is visible before it hurts" text="Population stability of benign scores — computed with no labels — tracks how far the false-positive rate drifts above its conformal budget." to="/robustness#drift" />
          <Finding accent="#f5b84b" big={meanRet != null ? fmt.pct(meanRet, 0) : '—'} title="Transfer keeps only a fraction" text="Mean share of in-dataset AUC retained when a detector trained on one capture is pointed at another network." to="/robustness#transfer" />
        </div>
        {best && summary?.titles && (
          <p className="mt-6 text-sm text-muted">Best mean rank across 3 datasets × 2 training regimes × seeds: <span className="font-medium text-text">{summary.titles[best] ?? best}</span> (Friedman, Nemenyi CD). <Link to="/benchmark" className="text-violet-soft hover:underline">See the full critical-difference diagram →</Link></p>
        )}
      </section>

      {/* ---------------------------------------------------------------- features */}
      <section className="mx-auto max-w-[1500px] px-6 pb-28">
        <div className="mb-10 max-w-2xl">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-violet-soft">The lab</div>
          <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Six ways to interrogate the results.</h2>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div key={f.to} initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: '-60px' }} transition={{ duration: 0.5, delay: (i % 3) * 0.07 }}>
              <Link to={f.to} className="focus-ring group block h-full">
                <Card className="relative h-full overflow-hidden p-6 transition-all duration-300 group-hover:-translate-y-1 group-hover:border-white/[0.16]">
                  <div className="pointer-events-none absolute -right-16 -top-16 h-44 w-44 rounded-full opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-40" style={{ background: f.color }} />
                  <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-xl" style={{ background: `${f.color}1f`, boxShadow: `inset 0 0 0 1px ${f.color}40`, color: f.color }}><f.icon size={19} /></div>
                  <h3 className="text-[17px] font-semibold tracking-tight">{f.title}</h3>
                  <p className="mt-2 text-[14px] leading-relaxed text-muted">{f.text}</p>
                  <div className="mt-5 flex items-center gap-1.5 text-[13px] font-medium" style={{ color: f.color }}>Open <ArrowRight size={14} className="transition-transform group-hover:translate-x-1" /></div>
                </Card>
              </Link>
            </motion.div>
          ))}
        </div>
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-4 px-6 py-8 text-sm text-faint">
          <div>Noctua · built by <a className="text-muted hover:text-text" href="https://www.linkedin.com/in/frangomezpulido" target="_blank" rel="noreferrer">Francisco Javier Gómez Pulido</a></div>
          <div className="flex gap-5"><Link className="hover:text-text" to="/about">Paper & method</Link><a className="hover:text-text" href={REPO} target="_blank" rel="noreferrer">GitHub</a><span>MIT license</span></div>
        </div>
      </footer>
    </div>
  )
}

function Finding({ big, title, text, accent, to }: { big: string; title: string; text: string; accent: string; to: string }) {
  return (
    <Link to={to} className="focus-ring group block">
      <Card className={cx('relative h-full overflow-hidden p-6 transition-all duration-300 group-hover:-translate-y-1')}>
        <div className="num text-[2.6rem] font-medium leading-none tracking-tight" style={{ color: accent, textShadow: `0 0 40px ${accent}55` }}>{big}</div>
        <h3 className="mt-5 text-[15px] font-semibold tracking-tight">{title}</h3>
        <p className="mt-2 text-[13.5px] leading-relaxed text-muted">{text}</p>
      </Card>
    </Link>
  )
}
