import { bisector } from 'd3-array'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line as d3line, area as d3area, curveMonotoneX } from 'd3-shape'
import { useMemo, useState } from 'react'
import { useSize } from '../../lib/hooks'

export interface Series {
  id: string
  label: string
  color: string
  points: [number, number][]
  band?: [number, number, number][] // x, lo, hi
  dashed?: boolean
  width?: number
  dim?: boolean
}

interface Props {
  series: Series[]
  xLabel?: string
  yLabel?: string
  xDomain?: [number, number]
  yDomain?: [number, number]
  xScale?: 'linear' | 'log'
  height?: number
  formatX?: (v: number) => string
  formatY?: (v: number) => string
  diagonal?: boolean
  xTicks?: number[]
  yTicks?: number[]
  hoverX?: boolean
  markers?: boolean
  refLines?: { y?: number; x?: number; label?: string; color?: string }[]
}

const M = { t: 12, r: 16, b: 38, l: 48 }
const defaultFmt = (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(0) : String(+v.toFixed(3)))

export function LineChart(p: Props) {
  const [ref, { w }] = useSize<HTMLDivElement>()
  const h = p.height ?? 300
  const [hover, setHover] = useState<number | null>(null)
  const fx = p.formatX ?? defaultFmt
  const fy = p.formatY ?? defaultFmt
  const iw = Math.max(10, w - M.l - M.r)
  const ih = h - M.t - M.b

  const { x, y, xs, ys } = useMemo(() => {
    const allX = p.series.flatMap((s) => s.points.map((q) => q[0]))
    const allY = p.series.flatMap((s) => s.points.map((q) => q[1]))
    const xd = p.xDomain ?? [Math.min(...allX), Math.max(...allX)]
    const yd = p.yDomain ?? [Math.min(0, ...allY), Math.max(...allY)]
    const x = p.xScale === 'log' ? scaleLog().domain(xd).range([0, iw]).clamp(true) : scaleLinear().domain(xd).range([0, iw])
    const y = scaleLinear().domain(yd).range([ih, 0]).nice()
    const xs = p.xTicks ?? (p.xScale === 'log' ? [1e-4, 1e-3, 1e-2, 1e-1, 1].filter((t) => t >= xd[0] && t <= xd[1]) : x.ticks(6))
    const ys = p.yTicks ?? y.ticks(5)
    return { x, y, xs, ys }
  }, [p.series, p.xDomain, p.yDomain, p.xScale, p.xTicks, p.yTicks, iw, ih])

  const path = d3line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1])).curve(curveMonotoneX).defined((d) => Number.isFinite(d[1]) && (p.xScale !== 'log' || d[0] > 0))
  const hoverXs = useMemo(() => Array.from(new Set(p.series.flatMap((s) => s.points.map((q) => q[0])))).sort((a, b) => a - b), [p.series])
  const bis = bisector<number, number>((d) => d).center

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    if (!p.hoverX) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const xv = p.xScale === 'log' ? (x as ReturnType<typeof scaleLog>).invert(px) : (x as ReturnType<typeof scaleLinear>).invert(px)
    setHover(hoverXs[bis(hoverXs, xv as number)])
  }

  return (
    <div ref={ref} className="relative w-full select-none">
      {w > 0 && (
        <svg width={w} height={h} role="img" aria-label={p.yLabel ?? 'chart'}>
          <g transform={`translate(${M.l},${M.t})`}>
            {ys.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${y(t)})`}>
                <line x2={iw} stroke="rgba(255,255,255,0.06)" />
                <text x={-10} dy="0.32em" textAnchor="end" className="num fill-faint text-[10px]">{fy(t)}</text>
              </g>
            ))}
            {xs.map((t) => (
              <g key={`x${t}`} transform={`translate(${x(t)},${ih})`}>
                <line y2={5} stroke="rgba(255,255,255,0.14)" />
                <text y={18} textAnchor="middle" className="num fill-faint text-[10px]">{fx(t)}</text>
              </g>
            ))}
            <line x2={iw} y1={ih} y2={ih} stroke="rgba(255,255,255,0.14)" />
            {p.diagonal && <line x1={x(x.domain()[0])} y1={y(y.domain()[0])} x2={x(x.domain()[1])} y2={y(y.domain()[1])} stroke="rgba(255,255,255,0.28)" strokeDasharray="4 4" />}
            {p.refLines?.map((r, i) =>
              r.y != null ? (
                <g key={i}>
                  <line x2={iw} y1={y(r.y)} y2={y(r.y)} stroke={r.color ?? 'rgba(255,255,255,0.3)'} strokeDasharray="3 4" />
                  {r.label && <text x={iw - 4} y={y(r.y) - 5} textAnchor="end" className="fill-faint text-[10px]">{r.label}</text>}
                </g>
              ) : (
                <g key={i}>
                  <line y2={ih} x1={x(r.x!)} x2={x(r.x!)} stroke={r.color ?? 'rgba(255,255,255,0.3)'} strokeDasharray="3 4" />
                  {r.label && <text x={x(r.x!) + 4} y={10} className="fill-faint text-[10px]">{r.label}</text>}
                </g>
              ),
            )}
            {p.series.map((s) => {
              const area = s.band ? d3area<[number, number, number]>().x((d) => x(d[0])).y0((d) => y(d[1])).y1((d) => y(d[2])).curve(curveMonotoneX) : null
              return (
                <g key={s.id} opacity={s.dim ? 0.25 : 1}>
                  {area && s.band && <path d={area(s.band) ?? ''} fill={s.color} opacity={0.14} />}
                  <path d={path(s.points) ?? ''} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinecap="round" strokeLinejoin="round" />
                  {p.markers && s.points.map((q, i) => <circle key={i} cx={x(q[0])} cy={y(q[1])} r={3} fill="#070a12" stroke={s.color} strokeWidth={1.6} />)}
                </g>
              )
            })}
            {hover != null && (
              <g>
                <line x1={x(hover)} x2={x(hover)} y2={ih} stroke="rgba(255,255,255,0.3)" />
                {p.series.map((s) => {
                  const q = s.points.find((pt) => pt[0] === hover)
                  return q ? <circle key={s.id} cx={x(hover)} cy={y(q[1])} r={4} fill={s.color} stroke="#070a12" strokeWidth={1.5} /> : null
                })}
              </g>
            )}
            <rect width={iw} height={ih} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
            {p.xLabel && <text x={iw / 2} y={ih + 34} textAnchor="middle" className="fill-faint text-[11px]">{p.xLabel}</text>}
            {p.yLabel && <text transform={`translate(${-36},${ih / 2}) rotate(-90)`} textAnchor="middle" className="fill-faint text-[11px]">{p.yLabel}</text>}
          </g>
        </svg>
      )}
      {hover != null && w > 0 && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-line bg-[#0b101c]/95 px-3 py-2 text-xs shadow-xl backdrop-blur"
          style={{ left: Math.min(w - 170, Math.max(0, M.l + x(hover) + 12)), top: M.t }}
        >
          <div className="num mb-1 text-muted">{p.xLabel ? `${p.xLabel.split(' ')[0]} ` : ''}{fx(hover)}</div>
          {p.series
            .map((s) => ({ s, q: s.points.find((pt) => pt[0] === hover) }))
            .filter((r) => r.q)
            .sort((a, b) => b.q![1] - a.q![1])
            .map(({ s, q }) => (
              <div key={s.id} className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                <span className="text-text">{s.label}</span>
                <span className="num ml-auto pl-3 text-muted">{fy(q![1])}</span>
              </div>
            ))}
        </div>
      )}
    </div>
  )
}
