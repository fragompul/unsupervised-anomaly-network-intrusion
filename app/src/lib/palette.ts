// Category colors tuned for additive blending on a near-black background.
const KNOWN: Record<string, string> = {
  normal: '#5b6a8f',
  dos: '#ff6b81', ddos: '#ff4d6d', generic: '#ff8fa3',
  probe: '#ffb84d', portscan: '#ffb84d', reconnaissance: '#ffc76b',
  r2l: '#c084fc', exploits: '#b38bff', webattack: '#d8a8ff',
  u2r: '#34d399', bruteforce: '#2dd4bf', backdoor: '#5eead4',
  fuzzers: '#60a5fa', bot: '#38bdf8', analysis: '#7dd3fc',
  shellcode: '#f472b6', worms: '#fb923c', infiltration: '#a3e635', heartbleed: '#facc15',
}
const FALLBACK = ['#ff6b81', '#ffb84d', '#c084fc', '#34d399', '#60a5fa', '#f472b6', '#fb923c', '#a3e635']

export function categoryColor(name: string, index = 0): string {
  return KNOWN[name.toLowerCase()] ?? FALLBACK[index % FALLBACK.length]
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
