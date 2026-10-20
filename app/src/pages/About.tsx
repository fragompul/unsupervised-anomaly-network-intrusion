import { Check, Copy, ExternalLink, FileText } from 'lucide-react'
import { useState } from 'react'
import { REPO } from '../App'
import { Card, PageHeader } from '../components/ui'
import { useData } from '../lib/context'

const BIBTEX = `@misc{gomezpulido2026noctua,
  title  = {Noctua: Unsupervised Network Intrusion Detection under
            Training Contamination, Temporal Drift and Dataset Shift},
  author = {G{\\'o}mez Pulido, Francisco Javier},
  year   = {2026},
  url    = {${REPO}}
}`

const COMMANDS = `pip install -e ".[dev,api]"
python -m ids_anomaly.cli data                      # verified download of 3 datasets
python -m ids_anomaly.cli benchmark --experiment all --seeds 5
python -m ids_anomaly.cli analyze                   # tables, stats, conformal coverage
python -m ids_anomaly.export                        # artifacts for this web app
cd app && npm install && npm run dev`

const CONTRIBUTIONS = [
  { t: 'A protocol, not a leaderboard', d: 'Fifteen unsupervised detectors on NSL-KDD, UNSW-NB15 and CIC-IDS2017 under clean and natural training, with seeds, bootstrap intervals, Friedman/Nemenyi tests and Holm-corrected comparisons.' },
  { t: 'Conformal guarantees, tested to breaking', d: 'Split-conformal p-values give any detector a finite-sample false-positive bound, and a way to see it fail. Averaged p-values merge heterogeneous detectors without losing validity.' },
  { t: 'Consensus-Trimmed Refit', d: 'A label-free defence against training contamination: three independent screeners vote which rows to drop before the base detector is fitted. Ablations over trim level, single screeners and an oracle trim show where it helps, what it costs on clean data, and why the committee guards against a bad screener.' },
  { t: 'Drift and transfer, measured', d: 'A week of real timestamps and a two-dataset harmonized feature space quantify how fast calibration decays, and show that a label-free PSI alarm anticipates it.' },
  { t: 'Explanations with a faithfulness test', d: 'Exact per-feature reconstruction attributions, checked by deletion curves and by whether attribution clusters recover attack families.' },
]

export default function About() {
  const { manifest } = useData()
  const [copied, setCopied] = useState<string | null>(null)
  const copy = (k: string, text: string) => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(k)
      setTimeout(() => setCopied(null), 1600)
    })
  }
  return (
    <div className="mx-auto max-w-[1100px] px-6 pb-24 pt-28">
      <PageHeader eyebrow="Paper & method" title={<>Unsupervised intrusion detection <span className="grad-text">under pressure.</span></>}>
        Noctua is the interactive companion of a research study on how unsupervised network intrusion detectors behave when the textbook assumptions (a clean training window, stationary traffic, the same network at deployment) stop being true.
      </PageHeader>

      <div className="mb-10 flex flex-wrap gap-3">
        <a href="./paper.pdf" className="focus-ring inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet to-[#5b8def] px-4 py-2.5 text-sm font-semibold text-white shadow-glow"><FileText size={15} /> Read the paper (PDF)</a>
        <a href={REPO} target="_blank" rel="noreferrer" className="focus-ring inline-flex items-center gap-2 rounded-xl border border-line-strong bg-white/[0.04] px-4 py-2.5 text-sm font-semibold hover:bg-white/[0.08]">Source code <ExternalLink size={14} /></a>
        <a href={`${REPO}/tree/main/docs`} target="_blank" rel="noreferrer" className="focus-ring inline-flex items-center gap-2 rounded-xl border border-line-strong bg-white/[0.04] px-4 py-2.5 text-sm font-semibold hover:bg-white/[0.08]">Documentation <ExternalLink size={14} /></a>
      </div>

      <h2 className="mb-4 text-xl font-semibold tracking-tight">Contributions</h2>
      <div className="mb-12 grid gap-3">
        {CONTRIBUTIONS.map((c, i) => (
          <Card key={c.t} className="flex gap-4 p-5">
            <span className="num flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet/15 text-sm text-violet-soft">{i + 1}</span>
            <div><h3 className="text-[15px] font-semibold">{c.t}</h3><p className="mt-1 text-[14px] leading-relaxed text-muted">{c.d}</p></div>
          </Card>
        ))}
      </div>

      <h2 className="mb-4 text-xl font-semibold tracking-tight">Pipeline</h2>
      <Card className="mb-12 overflow-x-auto p-6">
        <svg viewBox="0 0 960 190" className="min-w-[760px]" role="img" aria-label="Pipeline diagram">
          {[
            { x: 10, t: 'Datasets', s: 'NSL-KDD · UNSW-NB15 · CIC-IDS2017', c: '#60a5fa' },
            { x: 205, t: 'Regimes', s: 'clean · natural · injected ρ', c: '#22d3ee' },
            { x: 400, t: '15 detectors', s: 'fit without labels', c: '#8b7bff' },
            { x: 595, t: 'Conformal layer', s: 'p-values · merged ensembles', c: '#f472b6' },
            { x: 790, t: 'Evidence', s: 'AUC · FPR · drift · transfer', c: '#34d399' },
          ].map((n, i, a) => (
            <g key={n.t}>
              <rect x={n.x} y={50} width={160} height={90} rx={16} fill="rgba(255,255,255,0.04)" stroke={n.c} strokeOpacity={0.55} />
              <text x={n.x + 80} y={86} textAnchor="middle" className="fill-text text-[15px] font-semibold">{n.t}</text>
              <foreignObject x={n.x + 8} y={96} width={144} height={40}><div style={{ fontSize: 11, color: '#8f98b3', textAlign: 'center', lineHeight: 1.3 }}>{n.s}</div></foreignObject>
              {i < a.length - 1 && <path d={`M${n.x + 164},95 h27`} stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} markerEnd="url(#arr)" />}
            </g>
          ))}
          <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="rgba(255,255,255,0.55)" /></marker></defs>
          <text x={480} y={30} textAnchor="middle" className="fill-faint text-[11px] uppercase tracking-widest">labels are used only to score · a held-out clean slice calibrates thresholds</text>
          <text x={480} y={176} textAnchor="middle" className="fill-faint text-[11px]">each cell stores raw scores → analysis recomputes every table and figure</text>
        </svg>
      </Card>

      <h2 className="mb-4 text-xl font-semibold tracking-tight">Reproduce everything</h2>
      <Block text={COMMANDS} id="cmd" copied={copied} onCopy={copy} />

      <h2 className="mb-4 mt-12 text-xl font-semibold tracking-tight">Cite</h2>
      <Block text={BIBTEX} id="bib" copied={copied} onCopy={copy} />

      <p className="mt-10 text-xs text-faint">{manifest ? `Artifacts generated ${new Date(manifest.generated).toUTCString().slice(0, 16)} · Noctua v${manifest.version} · MIT license.` : ''} Datasets are © their respective authors (Tavallaee et al. 2009; Moustafa & Slay 2015; Sharafaldin et al. 2018) and are downloaded from public mirrors, not redistributed here.</p>
    </div>
  )
}

function Block({ text, id, copied, onCopy }: { text: string; id: string; copied: string | null; onCopy: (k: string, t: string) => void }) {
  return (
    <div className="glass relative overflow-hidden">
      <button onClick={() => onCopy(id, text)} className="focus-ring absolute right-3 top-3 flex items-center gap-1.5 rounded-lg border border-line bg-white/[0.05] px-2.5 py-1 text-xs text-muted hover:text-text" aria-label="Copy">
        {copied === id ? <><Check size={13} className="text-mint" /> Copied</> : <><Copy size={13} /> Copy</>}
      </button>
      <pre className="num overflow-x-auto p-5 pr-24 text-[12.5px] leading-relaxed text-[#c9cfe6]">{text}</pre>
    </div>
  )
}
