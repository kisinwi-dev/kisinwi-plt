import type { DriftLevel, SplitDriftInfo } from '../../../types/datasetComparison';

// Пороги интерпретации дрейфа на бэкенде:
// services/datasets/app/core/services/comparison.py, DRIFT_THRESHOLDS = (0.1, 0.25).
const DRIFT_THRESHOLDS_NOTE =
  'Пороги: < 0.1 — нет дрейфа, 0.1–0.25 — умеренный, > 0.25 — значительный.';

export const DRIFT_TOOLTIP =
  'Дрейф — насколько изменилось распределение изображений по классам между версиями. '
  + `Считается метриками JS и PSI. ${DRIFT_THRESHOLDS_NOTE}`;

export const JS_TOOLTIP =
  'JS-дивергенция (Дженсена–Шеннона) — симметричная мера различия двух распределений: '
  + `0 — совпадают, 1 — полностью различны. ${DRIFT_THRESHOLDS_NOTE}`;

export const PSI_TOOLTIP =
  'PSI (Population Stability Index) — мера сдвига распределения, чувствительна '
  + `к изменению долей отдельных классов. ${DRIFT_THRESHOLDS_NOTE}`;

export const DRIFT_LEVEL_LABELS: Record<DriftLevel, string> = {
  none: 'нет',
  moderate: 'умеренный',
  significant: 'значительный',
};

export const driftTone = (level: DriftLevel): 'good' | 'warn' | 'poor' =>
  level === 'none' ? 'good' : level === 'moderate' ? 'warn' : 'poor';

const LEVEL_SEVERITY: Record<DriftLevel, number> = { none: 0, moderate: 1, significant: 2 };

/** Худший уровень дрейфа по всем сплитам и обеим метрикам (JS и PSI); null — данных нет. */
export const worstDriftLevel = (drift: Record<string, SplitDriftInfo>): DriftLevel | null => {
  let worst: DriftLevel | null = null;
  for (const info of Object.values(drift)) {
    for (const level of [info.js_level, info.psi_level]) {
      if (level !== null && (worst === null || LEVEL_SEVERITY[level] > LEVEL_SEVERITY[worst])) {
        worst = level;
      }
    }
  }
  return worst;
};
