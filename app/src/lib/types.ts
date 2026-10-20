export interface DetectorMeta {
  id: string
  title: string
  family: string
}

export interface DatasetMeta {
  key: string
  title: string
  year: number
  citation: string
  n_points: number
  n_test: number
  n_features: number
  feature_names: string[]
  categories: string[]
  embeddings: ('pca' | 'umap')[]
  detectors: DetectorMeta[]
  score_scale: number
  feature_clip: number
  attack_fraction: number
  explain: {
    deletion: { k: number[]; attributed: number[]; random: number[] }
    attribution_cluster_nmi: number | null
    n_flagged_eval: number
  }
}

export interface Manifest {
  name: string
  version: string
  generated: string
  datasets: DatasetMeta[]
  replay: { n: number; detectors: string[] } | null
  detector_titles: Record<string, string>
}

export interface Curve {
  fpr: number[]
  tpr: number[]
  recall: number[]
  precision: number[]
}

export interface CurvesFile {
  curves: Record<string, Curve>
  coverage: Record<string, { alpha: number[]; fpr: number[] }>
}

export interface AEFile {
  layers: { w: number[][]; b: number[]; relu: boolean }[]
  cal_scores: number[]
  n_cal: number
  median_error_per_feature: number[]
}

export interface PointsFile {
  category: number[]
  attack: number[]
  label: string[] | null
  group?: number[]
  group_names?: string[]
}

export interface DatasetData {
  meta: DatasetMeta
  n: number
  embeddings: Partial<Record<'pca' | 'umap', Float32Array>>
  scores: Uint8Array // (nDetectors, n)
  features: Int8Array // (n, d)
  attack: Uint8Array
  category: Uint8Array
  group: Uint8Array | null
  groupNames: string[]
  attribution: Uint8Array // (n, 10): 5 feature indices then 5 shares
  ae: AEFile
  curves: CurvesFile
}

// --- analysis summary (subset of what the Python side writes) ------------------------------

export interface MainRow {
  dataset: string
  regime: 'clean' | 'natural'
  detector: string
  roc: number
  roc_std: number
  roc_ci?: [number, number]
  pr: number
  tpr_at_1pct_fpr: number
  a1_fpr: number
  a1_tpr: number
  a1_f1: number
  a5_fpr: number
  a5_tpr: number
  a5_f1: number
  fit_s?: number
  score_ms_per_1k?: number
  n_seeds: number
}

export interface FriedmanBlock {
  n_blocks: number
  statistic: number
  p: number
  cd: number
  mean_ranks: Record<string, number>
  cd_groups: string[][]
}

export interface Summary {
  titles: Record<string, string>
  main?: {
    table: MainRow[]
    stats: {
      all_seeds?: FriedmanBlock
      dataset_regime?: FriedmanBlock | null
      best_by_rank?: string
      holm_vs_best?: { method: string; mean_diff: number; p: number; p_holm: number; significant: boolean }[]
    }
    coverage: { dataset: string; regime: string; detector: string; alpha: number; fpr: number }[]
    categories: { dataset: string; regime: string; detector: string; category: string; n: number; recall_a5: number }[]
  }
  contamination?: {
    table: { dataset: string; rho: number; detector: string; roc: number; roc_std: number; pr: number; a1_fpr: number; a1_tpr: number; n_seeds: number }[]
    ctr_gain: { dataset: string; rho: number; detector: string; base: string; gain: number; gain_std: number; n: number }[]
    degradation: { dataset: string; detector: string; roc_clean: number; roc_5pct: number; drop: number }[]
  }
  drift?: {
    table: { detector: string; day: string; roc?: number; psi_benign: number; fpr_a1: number; fpr_a5: number; tpr_a5?: number; n_attack: number; n_normal: number }[]
    psi_fpr_spearman: { rho: number; p: number } | null
    drift_alarm: { threshold_psi: number; tp: number; fp: number; fn: number; precision: number | null; recall: number | null }
    drift_alarms?: { threshold_psi: number; tp: number; fp: number; fn: number; precision: number | null; recall: number | null }[]
    coverage: { detector: string; day: string; alpha: number; fpr: number }[]
  }
  transfer?: {
    table: { source: string; target: string; scaler: string; detector: string; roc: number; roc_std: number; pr: number; a1_fpr: number; a5_fpr: number; a5_tpr: number; n_seeds: number }[]
    retention: { source: string; target: string; scaler: string; detector: string; roc: number; in_dataset_roc: number | null; retention: number | null }[]
  }
}

export interface Replay {
  days: string[]
  detectors: { id: string; title: string }[]
  t: number[]
  attack: number[]
  day: number[]
  nlogp: Record<string, number[]>
  score_scale: number
}
