import { useSize } from '../../lib/hooks'
import { FAMILY_COLOR } from '../../lib/palette'

interface Props {
  ranks: Record<string, number>
  cd: number
  groups: string[][]
  title: (id: string) => string
  family: (id: string) => string
}

/** Demšar critical-difference diagram: methods on a rank axis, bars join methods that are not
 *  significantly different at the Nemenyi level. */
export function CDDiagram({ ranks, cd, groups, title, family }: Props) {
  const [ref, { w }] = useSize<HTMLDivElement>()
  const entries = Object.entries(ranks).sort((a, b) => a[1] - b[1])
  const n = entries.length
  const half = Math.ceil(n / 2)
  const left = entries.slice(0, half)
  const right = entries.slice(half).reverse()
  const pad = 190
  const axisY = 56
  const min = Math.floor(entries[0][1])
  const max = Math.ceil(entries[n - 1][1])
  const rowH = 22
  const h = axisY + 40 + Math.max(left.length, right.length) * rowH + groups.length * 12
  const sx = (r: number) => pad + ((r - min) / (max - min)) * Math.max(10, w - 2 * pad)
  return (
    <div ref={ref} className="w-full overflow-x-auto">
      {w > 0 && (
        <svg width={w} height={h}>
          <g transform={`translate(0,${axisY})`}>
            <line x1={sx(min)} x2={sx(max)} stroke="rgba(255,255,255,0.3)" />
            {Array.from({ length: max - min + 1 }, (_, i) => min + i).map((t) => (
              <g key={t} transform={`translate(${sx(t)},0)`}>
                <line y1={-5} y2={5} stroke="rgba(255,255,255,0.4)" />
                <text y={-12} textAnchor="middle" className="num fill-muted text-[11px]">{t}</text>
              </g>
            ))}
            {/* CD bar */}
            <g transform={`translate(${sx(min)},-34)`}>
              <line x2={sx(min + cd) - sx(min)} stroke="#f5b84b" strokeWidth={2} />
              <line y1={-4} y2={4} stroke="#f5b84b" strokeWidth={2} />
              <line x1={sx(min + cd) - sx(min)} x2={sx(min + cd) - sx(min)} y1={-4} y2={4} stroke="#f5b84b" strokeWidth={2} />
              <text x={(sx(min + cd) - sx(min)) / 2} y={-8} textAnchor="middle" className="num fill-amber text-[10.5px]">CD = {cd.toFixed(2)}</text>
            </g>
            {[...left.map((e, i) => ({ e, i, side: 'l' as const })), ...right.map((e, i) => ({ e, i, side: 'r' as const }))].map(({ e: [id, r], i, side }) => {
              const y = 22 + i * rowH
              const x = sx(r)
              const tx = side === 'l' ? pad - 16 : w - pad + 16
              return (
                <g key={id}>
                  <path d={`M${x},0 V${y} H${tx}`} fill="none" stroke="rgba(255,255,255,0.22)" />
                  <circle cx={x} cy={0} r={3} fill={FAMILY_COLOR[family(id)] ?? '#8f98b3'} />
                  <text x={tx + (side === 'l' ? -6 : 6)} y={y} dy="0.33em" textAnchor={side === 'l' ? 'end' : 'start'} className="fill-text text-[11.5px]">
                    {title(id)} <tspan className="num fill-faint">{r.toFixed(1)}</tspan>
                  </text>
                </g>
              )
            })}
            {groups.map((g, gi) => {
              const rs = g.map((id) => ranks[id])
              const y = 22 + Math.max(left.length, right.length) * rowH + 6 + gi * 10
              return <line key={gi} x1={sx(Math.min(...rs)) - 2} x2={sx(Math.max(...rs)) + 2} y1={y} y2={y} stroke="#8b7bff" strokeWidth={4} strokeLinecap="round" opacity={0.85} />
            })}
          </g>
        </svg>
      )}
    </div>
  )
}
