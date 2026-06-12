// Общие константы и хелперы страницы сравнения моделей.

import type { CompareMetric, CompareModelEntry, ModelMetrics } from '../../services/metricsService';
import { formatMetricValue } from '../../utils/format';

/** Сторона сравнения: модель с подписью и закреплённым цветом. */
export interface CompareSide {
  id: string;
  /** Однострочная подпись «имя · vN» — для вердикта, легенд и сносок. */
  label: string;
  /** Имя и версия раздельно — для шапок таблиц (рендерятся на разных строках). */
  name: string;
  version: string;
  color: string;
}

/**
 * Происхождение сохранённых весов модели: эпоха чекпоинта из всех эпох
 * обучения. Trainer сохраняет веса эпохи с лучшим значением early-stop-метрики
 * на валидационной выборке (по умолчанию — loss).
 */
export interface WeightsInfo {
  epoch: number;
  /** Early-stop-метрика (чистое имя; считается на валидационной выборке). */
  metric: string;
  /** Значение метрики на эпохе чекпоинта; null — улучшений не было, веса финальной эпохи. */
  value: number | null;
  totalEpochs: number;
}

/** Общее число эпох обучения — из длины кривых train/val (конфиг не хардкодим). */
export const totalEpochsOf = (metrics: ModelMetrics | null) =>
  Math.max(
    0,
    ...[...(metrics?.train ?? []), ...(metrics?.val ?? [])].map((m) => m.values.length),
  );

// Палитра сторон: цвет закреплён за позицией в списке (первая — базовая).
// Намеренно отличается от SPLIT_COLORS графиков ModelDetail, чтобы цвет
// модели не путался с цветом выборки.
export const COMPARE_COLORS = [
  '#5ea3b1', // teal — базовая
  '#c08a4f', // amber
  '#8a6fb1', // purple
  '#6fb178', // green
] as const;

/** Больше четырёх моделей — нечитаемые графики и таблица шире экрана. */
export const MAX_COMPARE_MODELS = 4;

/** Дельты меньше порога считаем нулевыми (шум float-арифметики). */
export const DELTA_EPSILON = 1e-9;

/**
 * Равенство с точностью отображения: значения, неразличимые в UI
 * (formatMetricValue показывает 4 знака), считаем ничьей — иначе звезда
 * у «лидера» с перевесом 1e-5 выглядит ошибкой рядом с двумя одинаковыми
 * числами.
 */
export const displayEqual = (a: number, b: number) =>
  formatMetricValue(a) === formatMetricValue(b);

/** Сравниваемое значение модели на test: одна точка, weights_value = final_value. */
export const primaryValue = (entry: CompareModelEntry) =>
  entry.weights_value ?? entry.final_value;

/**
 * Лидеры по метрике с учётом ничьих: все модели, чьё значение неотличимо
 * от лучшего с точностью отображения (displayEqual). Backend в best_model_id
 * при равенстве назначает лидером произвольную модель — поэтому считаем
 * по значениям сами. Пустой список — значения всех моделей совпадают,
 * лидерство не имеет смысла.
 */
export const metricLeaders = (metric: CompareMetric): string[] => {
  const values = metric.models.map(primaryValue);
  if (values.length === 0) return [];
  const best = metric.higher_is_better ? Math.max(...values) : Math.min(...values);
  const leaders = metric.models
    .filter((entry) => displayEqual(primaryValue(entry), best))
    .map((entry) => entry.model_id);
  return leaders.length === metric.models.length ? [] : leaders;
};
