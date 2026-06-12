// Реестр метрик обучения: описание (тултип), иконка и пороги качества.
// Набор имён фиксирован трейнером (services/trainer METRICS_REGISTRY + loss);
// неизвестные имена деградируют мягко: без тултипа/иконки, нейтральная окраска.

import { ICONS } from './icons';

export type MetricQuality = 'good' | 'fair' | 'poor';

export interface MetricMeta {
  description: string;
  icon: string;
  /** Пороги «значение ≥ x» для good/fair; нет — у метрики нет абсолютной шкалы (loss). */
  thresholds?: { good: number; fair: number };
  /** Меньше — лучше (loss); по умолчанию метрики растут к лучшему. */
  lowerIsBetter?: boolean;
}

const METRIC_META: Record<string, MetricMeta> = {
  accuracy: {
    description: 'Доля верных предсказаний среди всех примеров.',
    icon: ICONS.metricAccuracy,
    thresholds: { good: 0.85, fair: 0.6 },
  },
  precision: {
    description: 'Доля верных среди предсказанных положительных (macro по классам).',
    icon: ICONS.metricPrecision,
    thresholds: { good: 0.85, fair: 0.6 },
  },
  recall: {
    description: 'Полнота: доля найденных объектов каждого класса (macro).',
    icon: ICONS.metricRecall,
    thresholds: { good: 0.85, fair: 0.6 },
  },
  f1: {
    description: 'Гармоническое среднее precision и recall (macro).',
    icon: ICONS.metricF1,
    thresholds: { good: 0.85, fair: 0.6 },
  },
  auroc: {
    description: 'Площадь под ROC-кривой; 0.5 — случайный классификатор.',
    icon: ICONS.metricAuroc,
    thresholds: { good: 0.9, fair: 0.7 },
  },
  specificity: {
    description: 'Доля верно отвергнутых отрицательных примеров.',
    icon: ICONS.metricSpecificity,
    thresholds: { good: 0.85, fair: 0.6 },
  },
  kappa: {
    description: 'Каппа Коэна: согласие с истиной с поправкой на случайное угадывание.',
    icon: ICONS.metricKappa,
    // Шкала Ландиса–Коха: ≥0.6 — substantial, 0.4–0.6 — moderate.
    thresholds: { good: 0.6, fair: 0.4 },
  },
  loss: {
    description: 'Значение функции потерь — чем ниже, тем лучше; абсолютной шкалы качества нет.',
    icon: ICONS.metricLoss,
    lowerIsBetter: true,
  },
};

// Пояснения выборок (split) для подсказок в UI.
export const SPLIT_DESCRIPTIONS: Record<string, string> = {
  train: 'Обучающая выборка: на этих данных модель училась.',
  val: 'Валидационная выборка: контроль качества во время обучения, в само обучение не входит.',
  test: 'Test-выборка: отложенные данные для финальной оценки — модель не видела их при обучении.',
};

export const getMetricMeta = (name: string): MetricMeta | undefined =>
  METRIC_META[name.trim().toLowerCase()];

export const getMetricQuality = (
  meta: MetricMeta | undefined,
  value: number,
): MetricQuality | undefined => {
  if (!meta?.thresholds) return undefined;
  if (value >= meta.thresholds.good) return 'good';
  if (value >= meta.thresholds.fair) return 'fair';
  return 'poor';
};
