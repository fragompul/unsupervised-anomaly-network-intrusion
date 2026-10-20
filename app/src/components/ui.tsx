import { motion } from 'framer-motion'
import { Check, ChevronDown } from 'lucide-react'
import type { CSSProperties, ReactNode } from 'react'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

export const cx = (...a: (string | false | null | undefined)[]) => a.filter(Boolean).join(' ')

export function Card({ className, children, style }: { className?: string; children: ReactNode; style?: CSSProperties }) {
  return (
    <div className={cx('glass', className)} style={style}>
      {children}
    </div>
  )
}

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-3xl">
      {eyebrow && <div className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-violet-soft">{eyebrow}</div>}
      <h1 className="text-3xl font-semibold leading-[1.1] tracking-tight text-text sm:text-[2.6rem]">{title}</h1>
      {children && <p className="mt-4 text-[15px] leading-relaxed text-muted">{children}</p>}
    </header>
  )
}

export function Badge({ children, color = '#8b7bff', className }: { children: ReactNode; color?: string; className?: string }) {
  return (
    <span
      className={cx('inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium', className)}
      style={{ background: `${color}22`, color, boxShadow: `inset 0 0 0 1px ${color}40` }}
    >
      {children}
    </span>
  )
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  size = 'md',
  label,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: ReactNode; disabled?: boolean }[]
  size?: 'sm' | 'md'
  label?: string
}) {
  return (
    <div role="tablist" aria-label={label} className="no-scrollbar relative inline-flex max-w-full overflow-x-auto rounded-[12px] border border-line bg-white/[0.025] p-[3px]">
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={active}
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            className={cx(
              'focus-ring relative z-10 whitespace-nowrap rounded-[9px] font-medium transition-colors disabled:opacity-30',
              size === 'sm' ? 'px-2 py-1 text-xs' : 'px-3.5 py-1.5 text-[13px]',
              active ? 'text-white' : 'text-muted hover:text-text',
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${label ?? 'x'}`}
                className="absolute inset-0 -z-10 rounded-[9px] bg-gradient-to-b from-white/[0.14] to-white/[0.06] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)]"
                transition={{ type: 'spring', stiffness: 500, damping: 38 }}
              />
            )}
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Select({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string; group?: string }[]
  label?: string
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [box, setBox] = useState<{ left: number; width: number; top?: number; bottom?: number; maxH: number } | null>(null)
  const btn = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const id = useId()
  const selected = options.findIndex((o) => o.value === value)

  // Options are rendered in order of first appearance of each group so the keyboard index matches the DOM.
  const groups = Array.from(new Set(options.map((o) => o.group ?? '')))
  const ordered = groups.flatMap((g) => options.filter((o) => (o.group ?? '') === g))

  const place = useCallback(() => {
    const r = btn.current?.getBoundingClientRect()
    if (!r) return
    const below = window.innerHeight - r.bottom - 12
    const above = r.top - 12
    const up = below < 220 && above > below
    setBox({ left: r.left, width: Math.max(r.width, 190), maxH: Math.min(360, up ? above : below), ...(up ? { bottom: window.innerHeight - r.top + 6 } : { top: r.bottom + 6 }) })
  }, [])

  const close = useCallback((refocus = true) => {
    setOpen(false)
    if (refocus) btn.current?.focus()
  }, [])

  const choose = (i: number) => {
    onChange(ordered[i].value)
    close()
  }

  useEffect(() => {
    if (!open) return
    place()
    const outside = (e: MouseEvent) => {
      const t = e.target as Node
      if (!list.current?.contains(t) && !btn.current?.contains(t)) close(false)
    }
    const away = (e: Event) => { if (!list.current?.contains(e.target as Node)) close(false) }
    document.addEventListener('mousedown', outside)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', away, true)
    return () => {
      document.removeEventListener('mousedown', outside)
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', away, true)
    }
  }, [open, place, close])

  useEffect(() => {
    if (open) list.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: 'nearest' })
  }, [open, active])

  const openList = () => {
    setActive(Math.max(0, ordered.findIndex((o) => o.value === value)))
    setOpen(true)
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (!open) {
      if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(e.key)) { e.preventDefault(); openList() }
      return
    }
    if (e.key === 'Escape') { e.preventDefault(); close() }
    else if (e.key === 'ArrowDown') { e.preventDefault(); setActive((i) => Math.min(ordered.length - 1, i + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((i) => Math.max(0, i - 1)) }
    else if (e.key === 'Home') { e.preventDefault(); setActive(0) }
    else if (e.key === 'End') { e.preventDefault(); setActive(ordered.length - 1) }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(active) }
    else if (e.key === 'Tab') close(false)
    else if (e.key.length === 1) {
      const k = e.key.toLowerCase()
      const n = ordered.findIndex((o, i) => i > active && o.label.toLowerCase().startsWith(k))
      const m = n >= 0 ? n : ordered.findIndex((o) => o.label.toLowerCase().startsWith(k))
      if (m >= 0) setActive(m)
    }
  }

  let lastGroup: string | null = null
  return (
    <div className={cx('block', className)}>
      {label && <span id={`${id}-l`} className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-faint">{label}</span>}
      <button
        ref={btn}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={`${id}-list`}
        aria-labelledby={label ? `${id}-l` : undefined}
        onClick={() => (open ? close() : openList())}
        onKeyDown={onKey}
        className={cx(
          'focus-ring flex w-full items-center justify-between gap-2 rounded-[11px] border bg-white/[0.04] py-2 pl-3 pr-2.5 text-left text-[13px] text-text transition-colors hover:bg-white/[0.07]',
          open ? 'border-violet/60 bg-white/[0.07]' : 'border-line',
        )}
      >
        <span className="truncate">{options[selected]?.label ?? ''}</span>
        <ChevronDown size={15} className={cx('shrink-0 text-faint transition-transform duration-150', open && 'rotate-180 text-violet-soft')} />
      </button>
      {open && box && createPortal(
        <div
          ref={list}
          id={`${id}-list`}
          role="listbox"
          onKeyDown={onKey}
          style={{ position: 'fixed', left: box.left, width: box.width, top: box.top, bottom: box.bottom, maxHeight: box.maxH, zIndex: 100, colorScheme: 'dark' }}
          className="overflow-y-auto rounded-[14px] border border-line-strong bg-[#0c1120]/95 p-1.5 shadow-[0_24px_60px_-12px_rgba(0,0,0,0.85),0_0_0_1px_rgba(139,123,255,0.12)] backdrop-blur-xl"
        >
          {ordered.map((o, i) => {
            const g = o.group ?? ''
            const header = g !== lastGroup && g !== ''
            lastGroup = g
            const isSel = o.value === value
            return (
              <div key={o.value}>
                {header && <div className={cx('px-2.5 pb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-faint', i > 0 ? 'mt-1.5 pt-2' : 'pt-1')}>{g}</div>}
                <div
                  role="option"
                  aria-selected={isSel}
                  data-i={i}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => choose(i)}
                  className={cx(
                    'flex cursor-pointer items-center justify-between gap-2 rounded-lg px-2.5 py-1.5 text-[13px] transition-colors',
                    i === active ? 'bg-white/[0.09] text-white' : 'text-muted',
                    isSel && 'text-white',
                  )}
                >
                  <span className="truncate">{o.label}</span>
                  {isSel && <Check size={14} className="shrink-0 text-violet-soft" />}
                </div>
              </div>
            )
          })}
        </div>,
        document.body,
      )}
    </div>
  )
}

export function Slider({
  value, onChange, min, max, step, label, display,
}: { value: number; onChange: (v: number) => void; min: number; max: number; step: number; label: string; display: string }) {
  const fill = ((value - min) / (max - min)) * 100
  return (
    <label className="block">
      <span className="mb-1 flex items-baseline justify-between text-[11px] font-medium uppercase tracking-wider text-faint">
        {label}
        <span className="num text-[13px] normal-case tracking-normal text-text">{display}</span>
      </span>
      <input className="slider" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} style={{ ['--fill' as string]: `${fill}%` }} />
    </label>
  )
}

export function Stat({ label, value, sub, accent }: { label: string; value: ReactNode; sub?: ReactNode; accent?: string }) {
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wider text-faint">{label}</div>
      <div className="num mt-1 text-2xl font-medium" style={{ color: accent }}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-muted">{sub}</div>}
    </div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx('skeleton', className)} />
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex items-end justify-between gap-4">
      <h2 className="text-[15px] font-semibold tracking-tight text-text">{children}</h2>
      {hint && <div className="text-xs text-faint">{hint}</div>}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="flex h-full min-h-40 items-center justify-center rounded-xl border border-dashed border-line px-6 text-center text-sm text-faint">{children}</div>
}

export const fadeUp = {
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] as const },
}
