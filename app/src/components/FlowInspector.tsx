import { ArrowUpRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { featureZ, nlogp, pValue, topAttribution } from '../lib/data'
import { fmt } from '../lib/format'
import { CATEGORY_LABEL, FAMILY_COLOR, categoryColor } from '../lib/palette'
import type { DatasetData } from '../lib/types'
import { Badge } from './ui'

const cleanName = (n: string) => n.replace(/^(num|cat)__/, '').replace(/_/g, ' ')

export function FlowInspector({ ds, index, detector, alpha }: { ds: DatasetData; index: number; detector: string; alpha: number }) {
  const cat = ds.meta.categories[ds.category[index]]
  const attack = ds.attack[index] === 1
  const attrs = topAttribution(ds, index)
  const detIdx = ds.meta.detectors.findIndex((d) => d.id === detector)
  const nl = nlogp(ds, Math.max(0, detIdx), index)
  const p = pValue(nl)
  const flagged = p <= alpha
  const maxNl = 6
  const dets = ds.meta.detectors.map((d, i) => ({ ...d, nl: nlogp(ds, i, index) }))
  const topZ = Array.from({ length: ds.meta.n_features }, (_, j) => ({ j, z: featureZ(ds, index, j) }))
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
    .slice(0, 6)

  return (
    <div className="space-y-5">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge color={categoryColor(cat, ds.category[index])}>{CATEGORY_LABEL(cat)}</Badge>
          <Badge color={attack ? '#fb7185' : '#34d399'}>{attack ? 'ground truth: attack' : 'ground truth: normal'}</Badge>
          {ds.groupNames.length > 0 && ds.group && <Badge color="#60a5fa">{ds.groupNames[ds.group[index]]}</Badge>}
        </div>
        <div className="mt-3 flex items-end justify-between">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-wider text-faint">Conformal p-value</div>
            <div className="num text-3xl font-medium" style={{ color: flagged ? '#fb7185' : '#34d399' }}>{fmt.p(p)}</div>
          </div>
          <div className="text-right text-xs text-muted">
            <div className={flagged ? 'font-medium text-rose' : 'font-medium text-mint'}>{flagged ? 'FLAGGED' : 'passes'} @ α = {alpha.toFixed(3)}</div>
            <div className="mt-0.5">{ds.meta.detectors[detIdx]?.title}</div>
          </div>
        </div>
      </div>

      <section>
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-faint">All detectors · evidence (−log₁₀ p)</h3>
        <div className="space-y-1.5">
          {dets.map((d) => (
            <div key={d.id} className="grid grid-cols-[116px_1fr_34px] items-center gap-2 text-xs">
              <span className={d.id === detector ? 'truncate font-medium text-text' : 'truncate text-muted'}>{d.title}</span>
              <div className="relative h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                <div className="h-full rounded-full" style={{ width: `${Math.min(100, (d.nl / maxNl) * 100)}%`, background: FAMILY_COLOR[d.family] ?? '#8f98b3', opacity: d.id === detector ? 1 : 0.7 }} />
                <div className="absolute top-0 h-full w-px bg-white/70" style={{ left: `${(-Math.log10(alpha) / maxNl) * 100}%` }} />
              </div>
              <span className="num text-right text-muted">{d.nl.toFixed(1)}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-faint">White tick = current α. Bars beyond it flag the flow.</p>
      </section>

      <section>
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-faint">Why: autoencoder error attribution</h3>
        <div className="space-y-2">
          {attrs.map((a, r) => (
            <div key={r}>
              <div className="mb-0.5 flex items-baseline justify-between text-xs">
                <span className="truncate text-text">{cleanName(ds.meta.feature_names[a.feature])}</span>
                <span className="num text-muted">{(a.share * 100).toFixed(0)}% · z={featureZ(ds, index, a.feature).toFixed(1)}</span>
              </div>
              <div className="h-1.5 rounded-full bg-white/[0.06]">
                <div className="h-full rounded-full bg-gradient-to-r from-violet to-cyan" style={{ width: `${a.share * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[11px] text-faint">
          Exact decomposition of the score. Deleting the top-3 features removes {fmt.pct(ds.meta.explain.deletion.attributed[2], 0)} of the score on average vs {fmt.pct(ds.meta.explain.deletion.random[2], 0)} for random features.
        </p>
      </section>

      <section>
        <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-faint">Most extreme features (z-score)</h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          {topZ.map(({ j, z }) => (
            <div key={j} className="flex items-baseline justify-between text-xs">
              <span className="truncate pr-2 text-muted">{cleanName(ds.meta.feature_names[j])}</span>
              <span className="num" style={{ color: Math.abs(z) > 3 ? '#fb7185' : '#e8ebf5' }}>{z.toFixed(1)}</span>
            </div>
          ))}
        </div>
      </section>

      <Link to={`/playground?dataset=${ds.meta.key}&flow=${index}`} className="focus-ring glass-flat flex items-center justify-between px-3.5 py-2.5 text-[13px] font-medium text-text transition-colors hover:bg-white/[0.07]">
        Open this flow in the Playground <ArrowUpRight size={15} className="text-violet-soft" />
      </Link>
    </div>
  )
}
