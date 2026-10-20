export const FAMILY_OF: Record<string, string> = {
  iforest: 'tree', ocsvm: 'kernel', pca: 'reconstruction', gmm: 'density', knn: 'distance', lof: 'distance',
  ecod: 'density', copod: 'density', hbos: 'density', ae: 'reconstruction', deep_svdd: 'deep-one-class',
  vae: 'reconstruction', flow: 'density', neutral: 'self-supervised', mtt: 'self-supervised',
  ens_all: 'ensemble', ens_diverse: 'ensemble',
}
export const FAMILY_LABEL: Record<string, string> = {
  tree: 'Tree', kernel: 'Kernel', reconstruction: 'Reconstruction', density: 'Density', distance: 'Distance',
  'deep-one-class': 'Deep one-class', 'self-supervised': 'Self-supervised', ensemble: 'Ensemble', 'robust-wrapper': 'Robust wrapper',
}
const ENS: Record<string, string> = { ens_all: 'Ensemble · all', ens_diverse: 'Ensemble · diverse 5' }

export function familyOf(id: string): string {
  return FAMILY_OF[id.split('+')[0]] ?? 'other'
}

export function detTitle(id: string, titles: Record<string, string> = {}): string {
  if (ENS[id]) return ENS[id]
  const [base, rest] = id.split('+')
  return (titles[base] ?? base) + (rest ? ` +${rest}` : '')
}
