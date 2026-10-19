import { create } from 'zustand'

export type ColorBy = 'category' | 'score' | 'truth' | 'diff'
export type EmbeddingKind = 'pca' | 'umap'

interface ExplorerState {
  datasetKey: string
  embedding: EmbeddingKind
  detector: string
  colorBy: ColorBy
  alpha: number
  selected: number | null
  autoRotate: boolean
  hiddenCategories: string[]
  set: (patch: Partial<Omit<ExplorerState, 'set' | 'toggleCategory'>>) => void
  toggleCategory: (c: string) => void
}

export const useExplorer = create<ExplorerState>((set) => ({
  datasetKey: 'unsw_nb15',
  embedding: 'umap',
  detector: 'ae',
  colorBy: 'category',
  alpha: 0.05,
  selected: null,
  autoRotate: true,
  hiddenCategories: [],
  set: (patch) => set(patch),
  toggleCategory: (c) =>
    set((s) => ({
      hiddenCategories: s.hiddenCategories.includes(c) ? s.hiddenCategories.filter((x) => x !== c) : [...s.hiddenCategories, c],
    })),
}))
