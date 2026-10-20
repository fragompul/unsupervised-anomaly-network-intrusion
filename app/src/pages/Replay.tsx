import { AnimatePresence, motion } from 'framer-motion'
import { Pause, Play, RotateCcw } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Badge, Card, Empty, PageHeader, Segmented, Select, Skeleton, Slider } from '../components/ui'
import { loadReplay } from '../lib/data'
import { fmt } from '../lib/format'
import { useAsync, useSize } from '../lib/hooks'
import type { Replay as ReplayData } from '../lib/types'

const WINDOW = 450
const EVENTS_PER_SEC = 55

function psi(ref: number[], cur: number[], bins = 8): number {
  if (ref.length < 20 || cur.length < 20) return 0
  const sorted = [...ref].sort((a, b) => a - b)
  const edges = Array.from({ length: bins - 1 }, (_, i) => sorted[Math.floor(((i + 1) / bins) * sorted.length)])
  const hist = (arr: number[]) => {
    const h = new Array(bins).fill(0)
    for (const v of arr) { let k = 0; while (k < edges.length && v > edges[k]) k++; h[k]++ }
    return h.map((c) => Math.max(1e-4, c / arr.length))
  }
  const r = hist(ref), c = hist(cur)
  return r.reduce((s, ri, i) => s + (c[i] - ri) * Math.log(c[i] / ri), 0)
}

export default function Replay() {
  const { data: replay, loading, error } = useAsync<ReplayData>(loadReplay, [])
  if (loading) return <Shell><Skeleton className="h-[520px]" /></Shell>
  if (error || !replay) return <Shell><Empty>The replay stream is not available in this build.</Empty></Shell>
  return <Shell><Stream replay={replay} /></Shell>
}

function Stream({ replay }: { replay: ReplayData }) {
  const n = replay.t.length
  const [det, setDet] = useState(replay.detectors.find((d) => d.id === 'ae')?.id ?? replay.detectors[0].id)
  const [alpha, setAlpha] = useState(0.01)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState<'1' | '4' | '16'>('4')
  const [pos, setPos] = useState(0)
  const posRef = useRef(0)

  const nl = replay.nlogp[det]
  const flagged = useMemo(() => nl.map((v) => Math.pow(10, -v / replay.score_scale) <= alpha), [nl, alpha, replay.score_scale])

  // prefix sums for rolling windows
  const pre = useMemo(() => {
    const mk = () => new Int32Array(n + 1)
    const normals = mk(), fp = mk(), attacks = mk(), tp = mk(), alerts = mk()
    for (let i = 0; i < n; i++) {
      const a = replay.attack[i] === 1, f = flagged[i]
      normals[i + 1] = normals[i] + (a ? 0 : 1)
      fp[i + 1] = fp[i] + (!a && f ? 1 : 0)
      attacks[i + 1] = attacks[i] + (a ? 1 : 0)
      tp[i + 1] = tp[i] + (a && f ? 1 : 0)
      alerts[i + 1] = alerts[i] + (f ? 1 : 0)
    }
    return { normals, fp, attacks, tp, alerts }
  }, [n, flagged, replay.attack])

  const rolling = (i: number) => {
    const lo = Math.max(0, i - WINDOW)
    const nn = pre.normals[i] - pre.normals[lo], na = pre.attacks[i] - pre.attacks[lo]
    return { fpr: nn ? (pre.fp[i] - pre.fp[lo]) / nn : 0, tpr: na ? (pre.tp[i] - pre.tp[lo]) / na : 0, alerts: pre.alerts[i] - pre.alerts[lo], na }
  }

  // reference benign scores: first 40% of Tuesday
  const ref = useMemo(() => {
    const out: number[] = []
    for (let i = 0; i < n && replay.day[i] === 0; i++) if (replay.attack[i] === 0 && out.length < 600) out.push(nl[i] / replay.score_scale)
    return out
  }, [n, nl, replay])
  const psiSeries = useMemo(() => {
    const out: { i: number; v: number }[] = []
    for (let i = WINDOW; i <= n; i += 60) {
      const cur: number[] = []
      for (let k = i - WINDOW; k < i; k++) if (replay.attack[k] === 0) cur.push(nl[k] / replay.score_scale)
      out.push({ i, v: psi(ref, cur) })
    }
    return out
  }, [n, nl, ref, replay])
  const fprSeries = useMemo(() => Array.from({ length: Math.floor(n / 30) }, (_, k) => ({ i: k * 30 + 30, v: rolling(k * 30 + 30).fpr })), [n, pre]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    posRef.current = pos
  }, [pos])
  useEffect(() => {
    if (!playing) return
    let raf = 0, last = performance.now(), acc = 0
    const tick = (t: number) => {
      const dt = (t - last) / 1000
      last = t
      posRef.current = Math.min(n, posRef.current + dt * EVENTS_PER_SEC * +speed)
      acc += dt
      if (acc > 0.05) { acc = 0; setPos(posRef.current) }
      if (posRef.current >= n) { setPlaying(false); setPos(n); return }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, n])

  const i = Math.floor(pos)
  const r = rolling(i)
  const curPsi = useMemo(() => {
    if (i < 80) return 0
    const cur: number[] = []
    for (let k = Math.max(0, i - WINDOW); k < i; k++) if (replay.attack[k] === 0) cur.push(nl[k] / replay.score_scale)
    return psi(ref, cur)
  }, [i, nl, ref, replay])
  const day = replay.day[Math.min(n - 1, i)]
  const feed = useMemo(() => {
    const out: number[] = []
    for (let k = Math.min(n - 1, i - 1); k >= 0 && out.length < 9; k--) if (flagged[k]) out.push(k)
    return out
  }, [i, flagged, n])
  const dayStart = useMemo(() => replay.days.map((_, d) => replay.t[replay.day.indexOf(d)]), [replay])
  const clock = (k: number) => {
    const s = replay.t[k] - dayStart[replay.day[k]]
    return `${replay.days[replay.day[k]]} +${String(Math.floor(s / 3600)).padStart(2, '0')}:${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}`
  }
  const alarm = curPsi >= 0.25
  const fprBroken = r.fpr > alpha * 3

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-4">
        <button onClick={() => { if (pos >= n) { posRef.current = 0; setPos(0) } setPlaying(!playing) }} className="focus-ring flex h-10 items-center gap-2 rounded-xl bg-gradient-to-r from-violet to-[#5b8def] px-4 text-sm font-semibold text-white shadow-glow">
          {playing ? <Pause size={15} /> : <Play size={15} />} {playing ? 'Pause' : pos >= n ? 'Replay' : 'Play'}
        </button>
        <button onClick={() => { posRef.current = 0; setPos(0); setPlaying(true) }} className="focus-ring glass-flat flex h-10 w-10 items-center justify-center text-muted hover:text-text" aria-label="Restart"><RotateCcw size={15} /></button>
        <Segmented label="speed" value={speed} onChange={setSpeed} options={[{ value: '1', label: '1×' }, { value: '4', label: '4×' }, { value: '16', label: '16×' }]} />
        <Select className="w-52" label="Detector" value={det} onChange={setDet} options={replay.detectors.map((d) => ({ value: d.id, label: d.title }))} />
        <div className="w-56"><Slider label="Alert budget α" value={Math.log10(alpha)} min={-3} max={-1} step={0.01} display={`${(alpha * 100).toFixed(alpha < 0.01 ? 2 : 1)}%`} onChange={(v) => setAlpha(Math.pow(10, v))} /></div>
        <div className="ml-auto text-right"><div className="text-[11px] uppercase tracking-wider text-faint">Now replaying</div><div className="num text-xl">{replay.days[day]} <span className="text-faint">· event {fmt.int(i)} / {fmt.int(n)}</span></div></div>
      </div>

      <Card className="p-5"><Timeline replay={replay} flagged={flagged} pos={pos} onSeek={(k) => { posRef.current = k; setPos(k) }} /></Card>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1fr_1fr_340px]">
        <Card className="p-5">
          <div className="mb-1 flex items-center justify-between"><h3 className="text-[13px] font-semibold">Rolling false-positive rate</h3><Badge color={fprBroken ? '#fb7185' : '#34d399'}>{fprBroken ? 'guarantee broken' : 'within budget'}</Badge></div>
          <div className="num mb-2 text-3xl" style={{ color: fprBroken ? '#fb7185' : '#34d399' }}>{fmt.pct(r.fpr, 2)} <span className="text-sm text-faint">vs nominal {fmt.pct(alpha, 2)}</span></div>
          <Trace data={fprSeries} pos={i} color={fprBroken ? '#fb7185' : '#34d399'} ref={alpha} max={Math.max(alpha * 8, 0.05)} days={replay.day} />
        </Card>
        <Card className="p-5">
          <div className="mb-1 flex items-center justify-between"><h3 className="text-[13px] font-semibold">Label-free drift monitor (PSI)</h3><AnimatePresence>{alarm && <motion.span initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ opacity: 0 }}><Badge color="#fb7185">DRIFT ALARM</Badge></motion.span>}</AnimatePresence></div>
          <div className="num mb-2 text-3xl" style={{ color: alarm ? '#fb7185' : curPsi > 0.1 ? '#f5b84b' : '#34d399' }}>{curPsi.toFixed(2)} <span className="text-sm text-faint">alarm ≥ 0.25</span></div>
          <Trace data={psiSeries} pos={i} color={alarm ? '#fb7185' : '#8b7bff'} ref={0.25} max={Math.max(0.6, ...psiSeries.map((p) => p.v))} days={replay.day} />
        </Card>
        <Card className="p-5">
          <h3 className="mb-3 text-[13px] font-semibold">Recent alerts</h3>
          <div className="space-y-1.5">
            <AnimatePresence initial={false}>
              {feed.map((k) => {
                const atk = replay.attack[k] === 1
                return (
                  <motion.div key={k} layout initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="flex items-center gap-2.5 rounded-lg border border-line bg-white/[0.025] px-2.5 py-1.5 text-xs">
                    <span className="h-2 w-2 rounded-full" style={{ background: atk ? '#fb7185' : '#8b7bff' }} />
                    <span className="num text-muted">{clock(k)}</span>
                    <span className="flex-1" />
                    <Badge color={atk ? '#fb7185' : '#8b7bff'}>{atk ? 'true attack' : 'false alarm'}</Badge>
                  </motion.div>
                )
              })}
            </AnimatePresence>
            {!feed.length && <div className="py-6 text-center text-xs text-faint">No alerts yet</div>}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-3">
            <div><div className="text-[10px] uppercase tracking-wider text-faint">Window recall</div><div className="num text-lg text-mint">{r.na ? fmt.pct(r.tpr, 0) : 'n/a'}</div></div>
            <div><div className="text-[10px] uppercase tracking-wider text-faint">Alerts / window</div><div className="num text-lg">{r.alerts}</div></div>
          </div>
        </Card>
      </div>
      <p className="max-w-3xl text-xs leading-relaxed text-faint">Stream: 6,000 CIC-IDS2017 flows (1,500 per day, Tuesday → Friday, in time order). Scores are conformal p-values calibrated on <em>Monday’s</em> benign traffic, so the alert budget is a promise made on Monday. The PSI monitor compares the benign scores in the last {WINDOW} events with early Tuesday and uses no labels.</p>
    </div>
  )
}

function Timeline({ replay, flagged, pos, onSeek }: { replay: ReplayData; flagged: boolean[]; pos: number; onSeek: (k: number) => void }) {
  const [wrap, { w }] = useSize<HTMLDivElement>()
  const canvas = useRef<HTMLCanvasElement>(null)
  const n = replay.t.length
  const H = 86
  useEffect(() => {
    const c = canvas.current
    if (!c || !w) return
    const dpr = window.devicePixelRatio || 1
    c.width = w * dpr
    c.height = H * dpr
    const g = c.getContext('2d')!
    g.scale(dpr, dpr)
    g.clearRect(0, 0, w, H)
    const sx = (k: number) => (k / n) * w
    // day bands
    for (let d = 0; d < replay.days.length; d++) {
      const a = replay.day.indexOf(d)
      g.fillStyle = d % 2 ? 'rgba(255,255,255,0.025)' : 'rgba(255,255,255,0.0)'
      g.fillRect(sx(a), 0, sx(a + n / 4) - sx(a), H)
      g.fillStyle = 'rgba(143,152,179,0.9)'
      g.font = '11px Inter Variable, sans-serif'
      g.fillText(replay.days[d], sx(a) + 6, 14)
    }
    const k = Math.floor(pos)
    for (let e = 0; e < n; e++) {
      const x = sx(e)
      const atk = replay.attack[e] === 1
      const f = flagged[e]
      const seen = e < k
      let col = 'rgba(93,102,132,0.35)'
      let y = 62, h = 8
      if (atk && f) { col = '#fb7185'; y = 22; h = 30 }
      else if (atk) { col = '#f5b84b'; y = 40; h = 14 }
      else if (f) { col = '#8b7bff'; y = 24; h = 26 }
      if (!seen) col = col.startsWith('#') ? `${col}33` : 'rgba(93,102,132,0.12)'
      g.fillStyle = col
      g.fillRect(x, y, Math.max(1, w / n), h)
    }
    g.fillStyle = '#fff'
    g.fillRect(sx(k) - 1, 6, 2, H - 10)
  }, [w, pos, flagged, n, replay])
  return (
    <div ref={wrap} className="relative w-full">
      <canvas ref={canvas} style={{ width: '100%', height: H, cursor: 'pointer' }} onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onSeek(((e.clientX - r.left) / r.width) * n) }} />
      <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-muted">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-1 rounded-sm bg-[#fb7185]" />caught attack</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-1 rounded-sm bg-[#f5b84b]" />missed attack</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-1 rounded-sm bg-[#8b7bff]" />false alarm</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-1 rounded-sm bg-[#5d6684]" />normal · click to seek</span>
      </div>
    </div>
  )
}

function Trace({ data, pos, color, ref: refLine, max, days }: { data: { i: number; v: number }[]; pos: number; color: string; ref: number; max: number; days: number[] }) {
  const [wrap, { w }] = useSize<HTMLDivElement>()
  const h = 130
  const n = days.length
  const x = (i: number) => (i / n) * w
  const y = (v: number) => h - 6 - Math.min(1, v / max) * (h - 14)
  const shown = data.filter((d) => d.i <= pos)
  const path = (arr: { i: number; v: number }[]) => arr.map((d, k) => `${k ? 'L' : 'M'}${x(d.i).toFixed(1)},${y(d.v).toFixed(1)}`).join(' ')
  return (
    <div ref={wrap} className="w-full">
      {w > 0 && (
        <svg width={w} height={h}>
          {[1, 2, 3].map((d) => <line key={d} x1={x(days.indexOf(d))} x2={x(days.indexOf(d))} y2={h} stroke="rgba(255,255,255,0.08)" />)}
          <line x1={0} x2={w} y1={y(refLine)} y2={y(refLine)} stroke="rgba(255,255,255,0.4)" strokeDasharray="4 4" />
          <path d={path(data)} fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth={1.5} />
          <path d={path(shown)} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
          <line x1={x(pos)} x2={x(pos)} y2={h} stroke="rgba(255,255,255,0.7)" />
        </svg>
      )}
    </div>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-[1500px] px-6 pb-24 pt-28">
      <PageHeader eyebrow="Live replay" title={<>Watch the guarantee <span className="grad-text">break in real time.</span></>}>
        Four days of traffic as a stream, scored against a threshold fixed on Monday. The false-positive rate leaves its budget on some days, and a label-free monitor sees it coming.
      </PageHeader>
      {children}
    </div>
  )
}
