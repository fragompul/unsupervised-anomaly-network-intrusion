export const fmt = {
  auc: (v: number | undefined | null) => (v == null || Number.isNaN(v) ? '—' : v.toFixed(3)),
  pct: (v: number | undefined | null, d = 1) => (v == null || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(d)}%`),
  int: (v: number) => v.toLocaleString('en-US'),
  compact: (v: number) => Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(v),
  p: (p: number) => (p < 0.001 ? p.toExponential(1).replace('e-', '×10⁻').replace(/⁻(\d)/, (_, d) => '⁻' + '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]) : p.toFixed(p < 0.1 ? 3 : 2)),
  secs: (v: number | undefined) => (v == null ? '—' : v < 1 ? `${(v * 1000).toFixed(0)} ms` : v < 100 ? `${v.toFixed(1)} s` : `${(v / 60).toFixed(1)} min`),
  signed: (v: number, d = 3) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`,
}

export const DATASET_ORDER = ['nsl_kdd', 'unsw_nb15', 'cicids2017']
export const DATASET_SHORT: Record<string, string> = { nsl_kdd: 'NSL-KDD', unsw_nb15: 'UNSW-NB15', cicids2017: 'CIC-IDS2017' }
