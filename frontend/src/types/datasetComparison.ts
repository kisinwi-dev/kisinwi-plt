// Типы сравнения версий датасета — зеркало services/datasets/app/api/schemas/comparison.py.

export type DriftLevel = 'none' | 'moderate' | 'significant';

/** Изменение значения между двумя версиями. */
export interface ValueDelta {
  from_value: number;
  to_value: number;
  /** Разница (to - from). */
  delta: number;
  /** Изменение в процентах от базового значения (null, если базовое значение 0). */
  percent_change: number | null;
}

/** Общие поля всех ответов сравнения версий. */
export interface ComparisonBase {
  dataset_id: string;
  from_version_id: string;
  to_version_id: string;
}

/** Сравнение количества изображений между версиями. */
export interface CountsComparison extends ComparisonBase {
  num_samples: ValueDelta;
  added_splits: string[];
  removed_splits: string[];
  per_split: Record<string, ValueDelta>;
  /** split -> class -> дельта. */
  per_class: Record<string, Record<string, ValueDelta>>;
}

/** Изменения состава классов одного сплита. */
export interface ClassChanges {
  added_classes: string[];
  removed_classes: string[];
  common_classes: string[];
}

/** Drift-метрики распределения классов одного сплита (null при пустом сплите). */
export interface SplitDriftInfo {
  js_divergence: number | null;
  js_level: DriftLevel | null;
  psi: number | null;
  psi_level: DriftLevel | null;
}

/** Сравнение распределений классов между версиями (ключи — сплиты). */
export interface DistributionComparison extends ComparisonBase {
  class_changes: Record<string, ClassChanges>;
  drift: Record<string, SplitDriftInfo>;
}

/** Сравнение баланса классов между версиями. */
export interface BalanceComparison extends ComparisonBase {
  overall_balance: ValueDelta;
  per_split: Record<string, ValueDelta>;
}

/** Сравнение размеров и форматов изображений между версиями. */
export interface SizeStatsComparison extends ComparisonBase {
  image_format_stats: Record<string, ValueDelta>;
  /** split -> "WxH" -> дельта. */
  size_counts_per_split: Record<string, Record<string, ValueDelta>>;
}

/** Счётчики по-файлового diff. */
export interface FilesDiffSummary {
  added_count: number;
  removed_count: number;
  common_count: number;
}

/** По-файловый diff между версиями (пути вида train/cat/img.jpg). */
export interface FilesDiffResponse extends ComparisonBase, FilesDiffSummary {
  added: string[];
  removed: string[];
}

/** Полная сводка сравнения двух версий датасета. */
export interface VersionComparisonResponse extends ComparisonBase {
  counts: CountsComparison;
  distribution: DistributionComparison;
  balance: BalanceComparison;
  size_stats: SizeStatsComparison;
  /** Только счётчики; списки файлов — отдельным запросом /compare/files. */
  files: FilesDiffSummary;
}
