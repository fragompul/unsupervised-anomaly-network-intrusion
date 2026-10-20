// Category colours: 'normal' recedes; attack families get maximally distinct hues (assigned by
// manifest order so a legend never repeats a colour), tuned for additive blending on near-black.
const NORMAL = '#5b6a8f'
const PALETTE = ['#ff5d73', '#ffb020', '#a78bfa', '#2dd4bf', '#60a5fa', '#f472b6', '#a3e635', '#fb923c', '#22d3ee', '#e879f9', '#facc15', '#34d399']

export function categoryColor(name: string, index = 0): string {
  if (name.toLowerCase() === 'normal') return NORMAL
  // index 0 is 'normal' in every manifest, so attacks start at index 1
  return PALETTE[(Math.max(1, index) - 1) % PALETTE.length]
}

export const CATEGORY_LABEL = (c: string) =>
  ({ dos: 'DoS', ddos: 'DDoS', r2l: 'R2L', u2r: 'U2R', webattack: 'Web attack', portscan: 'Port scan', bruteforce: 'Brute force' } as Record<string, string>)[c] ??
  c.charAt(0).toUpperCase() + c.slice(1)

export function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16)
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255]
}

// Sequential ramp for anomaly strength: deep indigo → violet → magenta → amber (colour-blind safe
// because lightness increases monotonically with value).
const RAMP: [number, [number, number, number]][] = [
  [0, [38, 48, 92]],
  [0.35, [112, 94, 255]],
  [0.65, [236, 90, 190]],
  [1, [255, 214, 102]],
]
export function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t))
  for (let i = 1; i < RAMP.length; i++) {
    if (x <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1]
      const [t1, c1] = RAMP[i]
      const f = (x - t0) / (t1 - t0)
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f]
    }
  }
  return RAMP[RAMP.length - 1][1]
}
export const rampCss = (t: number) => {
  const [r, g, b] = ramp(t)
  return `rgb(${r | 0},${g | 0},${b | 0})`
}

// Detector families → stable hues used in every chart.
export const FAMILY_COLOR: Record<string, string> = {
  tree: '#34d399', kernel: '#60a5fa', density: '#f5b84b', distance: '#22d3ee', reconstruction: '#8b7bff',
  'deep-one-class': '#f472b6', 'self-supervised': '#fb7185', ensemble: '#ffffff', 'robust-wrapper': '#a3e635', other: '#8f98b3',
}

// Stable per-detector hues for multi-line charts (families repeat, detectors must be separable).
export const DET_COLOR: Record<string, string> = {
  ae: '#8b7bff', vae: '#c4bcff', knn: '#22d3ee', iforest: '#34d399', gmm: '#f5b84b', pca: '#60a5fa',
  ecod: '#fb923c', deep_svdd: '#f472b6', flow: '#fb7185', neutral: '#a3e635', mtt: '#e879f9', lof: '#2dd4bf',
  ocsvm: '#93c5fd', copod: '#fdba74', hbos: '#fcd34d', ens_all: '#ffffff', ens_diverse: '#d1d5db',
}
export const detColor = (id: string) => DET_COLOR[id.split('+')[0]] ?? '#8f98b3'
