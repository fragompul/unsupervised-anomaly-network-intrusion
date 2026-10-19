import { motion } from 'framer-motion'
import type { CSSProperties, ReactNode } from 'react'

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
    <div role="tablist" aria-label={label} className="relative inline-flex rounded-[12px] border border-line bg-white/[0.025] p-[3px]">
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
              'focus-ring relative z-10 rounded-[9px] font-medium transition-colors disabled:opacity-30',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-1.5 text-[13px]',
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
  const groups = Array.from(new Set(options.map((o) => o.group ?? '')))
  return (
    <label className={cx('block', className)}>
      {label && <span className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-faint">{label}</span>}
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="focus-ring w-full appearance-none rounded-[11px] border border-line bg-white/[0.04] py-2 pl-3 pr-8 text-[13px] text-text transition-colors hover:bg-white/[0.07]"
        >
          {groups.map((g) =>
            g ? (
              <optgroup key={g} label={g}>
                {options.filter((o) => o.group === g).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </optgroup>
            ) : (
              options.filter((o) => !o.group).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)
            ),
          )}
        </select>
        <svg className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m6 8 4 4 4-4" strokeLinecap="round" strokeLinejoin="round" /></svg>
      </div>
    </label>
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
