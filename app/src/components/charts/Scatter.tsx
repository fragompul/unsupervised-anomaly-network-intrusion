import { scaleLinear, scaleLog } from 'd3-scale'
import { useState } from 'react'
import { useSize } from '../../lib/hooks'

export interface Dot { id: string; label: string; x: number; y: number; color: string; r?: number }

export function Scatter({ dots, xLabel, yLabel, height = 340, fx = (v: number) => String(v), fy = (v: number) => v.toFixed(2), xLog = false, xTicks }: {
  dots: Dot[]; xLabel: string; yLabel: string; height?: number; fx?: (v: number) => string; fy?: (v: number) => string; xLog?: boolean; xTicks?: number[]
}) {
  const [ref, { w }] = useSize<HTMLDivElement>()
  const [hov, setHov] = useState<string | null>(null)
  const M = { t: 14, r: 20, b: 40, l: 50 }
  const iw = Math.max(10, w - M.l - M.r)
  const ih = height - M.t - M.b
  const xs = dots.map((d) => d.x)
  const ys = dots.map((d) => d.y)
  const x = xLog ? scaleLog().domain([Math.min(...xs) * 0.7, Math.max(...xs) * 1.4]).range([0, iw]) : scaleLinear().domain([Math.min(...xs), Math.max(...xs)]).nice().range([0, iw])
  const y = scaleLinear().domain([Math.min(...ys) - 0.01, Math.max(...ys) + 0.01]).nice().range([ih, 0])
  const xt = xTicks ?? x.ticks(5)
  return (
    <div ref={ref} className="relative w-full">
      {w > 0 && (
        <svg width={w} height={height}>
          <g transform={`translate(${M.l},${M.t})`}>
            {y.ticks(5).map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x2={iw} stroke="rgba(255,255,255,0.06)" />
                <text x={-10} dy="0.32em" textAnchor="end" className="num fill-faint text-[10px]">{fy(t)}</text>
              </g>
            ))}
            {xt.map((t) => (
              <g key={t} transform={`translate(${x(t)},${ih})`}>
                <line y2={5} stroke="rgba(255,255,255,0.14)" />
                <text y={18} textAnchor="middle" className="num fill-faint text-[10px]">{fx(t)}</text>
              </g>
            ))}
            <line x2={iw} y1={ih} y2={ih} stroke="rgba(255,255,255,0.14)" />
            {dots.map((d) => (
              <g key={d.id} transform={`translate(${x(d.x)},${y(d.y)})`} onPointerEnter={() => setHov(d.id)} onPointerLeave={() => setHov(null)}>
                <circle r={(d.r ?? 6) + 6} fill={d.color} opacity={hov === d.id ? 0.22 : 0.1} />
                <circle r={d.r ?? 6} fill={d.color} stroke="#070a12" strokeWidth={1.5} opacity={hov && hov !== d.id ? 0.4 : 1} />
                <text x={(d.r ?? 6) + 5} y={4} className={hov === d.id ? 'fill-text text-[11px] font-medium' : 'fill-muted text-[10.5px]'}>{d.label}</text>
              </g>
            ))}
            <text x={iw / 2} y={ih + 34} textAnchor="middle" className="fill-faint text-[11px]">{xLabel}</text>
            <text transform={`translate(${-38},${ih / 2}) rotate(-90)`} textAnchor="middle" className="fill-faint text-[11px]">{yLabel}</text>
          </g>
        </svg>
      )}
    </div>
  )
}
