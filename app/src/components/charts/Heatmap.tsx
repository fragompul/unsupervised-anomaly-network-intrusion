import { useState } from 'react'
import { useSize } from '../../lib/hooks'

interface Props {
  rows: string[]
  cols: string[]
  values: (number | null)[][]
  color: (v: number) => string
  format?: (v: number) => string
  rowLabel?: (r: string) => string
  colLabel?: (c: string) => string
  cellH?: number
  labelW?: number
  textColor?: (v: number) => string
  onCell?: (r: number, c: number) => void
}

export function Heatmap({ rows, cols, values, color, format = (v) => v.toFixed(2), rowLabel = (r) => r, colLabel = (c) => c, cellH = 30, labelW = 132, textColor, onCell }: Props) {
  const [ref, { w }] = useSize<HTMLDivElement>()
  const [hov, setHov] = useState<[number, number] | null>(null)
  const headH = 46
  const cw = Math.max(36, (w - labelW) / Math.max(1, cols.length))
  return (
    <div ref={ref} className="w-full overflow-x-auto">
      {w > 0 && (
        <svg width={Math.max(w, labelW + cw * cols.length)} height={headH + rows.length * cellH}>
          {cols.map((c, j) => (
            <text key={c} transform={`translate(${labelW + j * cw + cw / 2},${headH - 8}) rotate(-28)`} className="fill-muted text-[10.5px]">{colLabel(c)}</text>
          ))}
          {rows.map((r, i) => (
            <g key={r} transform={`translate(0,${headH + i * cellH})`}>
              <text x={labelW - 10} y={cellH / 2} dy="0.34em" textAnchor="end" className={hov?.[0] === i ? 'fill-text text-[12px]' : 'fill-muted text-[12px]'}>{rowLabel(r)}</text>
              {cols.map((c, j) => {
                const v = values[i]?.[j]
                return (
                  <g key={c} onPointerEnter={() => setHov([i, j])} onPointerLeave={() => setHov(null)} onClick={() => onCell?.(i, j)} style={{ cursor: onCell ? 'pointer' : 'default' }}>
                    <rect x={labelW + j * cw + 1.5} y={1.5} width={cw - 3} height={cellH - 3} rx={6} fill={v == null ? 'rgba(255,255,255,0.03)' : color(v)} stroke={hov?.[0] === i && hov?.[1] === j ? 'rgba(255,255,255,0.8)' : 'transparent'} />
                    {v != null && (
                      <text x={labelW + j * cw + cw / 2} y={cellH / 2} dy="0.34em" textAnchor="middle" className="num text-[10.5px]" fill={textColor ? textColor(v) : 'rgba(255,255,255,0.92)'}>{format(v)}</text>
                    )}
                  </g>
                )
              })}
            </g>
          ))}
        </svg>
      )}
    </div>
  )
}
