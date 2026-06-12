// Типы и расчётные хелперы графиков кривых по эпохам — отдельно от
// компонентов (curveChart.tsx), чтобы не ломать react fast refresh.

/** Серия графика: одна кривая с подписью и закреплённым цветом. */
export interface CurveSeries {
  /** Ключ значения в строках rows. */
  key: string;
  label: string;
  color: string;
  /** Пояснение серии (тултип в легенде). */
  hint?: string;
  /**
   * Эпоха сохранённых весов, относящаяся к этой серии, — для сводки
   * zoom-оверлея («веса: эпоха N»). Вертикаль на графике — checkpoints.
   */
  checkpointEpoch?: number | null;
}

/** Пунктирная вертикаль эпохи сохранённых весов. */
export interface CheckpointMark {
  epoch: number;
  color: string;
  /** Серия, к которой относится чекпоинт, — приглушается вместе с ней. */
  seriesKey?: string;
}

export type EpochRow = { epoch: number } & Partial<Record<string, number>>;

/** Данные одного графика: метрика и значения серий по эпохам. */
export interface EpochChart {
  /** Имя метрики (loss, accuracy, …) — заголовок и ключ порядка. */
  name: string;
  rows: EpochRow[];
  maxPoints: number;
  domain: [number, number];
}

/** Один график сетки с его сериями и чекпоинтами. */
export interface CurveGridItem {
  chart: EpochChart;
  series: CurveSeries[];
  checkpoints: CheckpointMark[];
}

// Ось Y от нуля «прижимает» кривые вроде accuracy (0.90–0.95) к потолку графика,
// поэтому домен считаем по фактическому диапазону данных с запасом 8% сверху и снизу.
export const yDomain = (rows: EpochRow[], keys: string[]): [number, number] => {
  let min = Infinity;
  let max = -Infinity;
  for (const row of rows) {
    for (const key of keys) {
      const value = row[key];
      if (value === undefined) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  const padding = (max - min || Math.abs(max) || 1) * 0.08;
  return [min - padding, max + padding];
};

/** Лучшее (по направлению метрики) и финальное значение одной кривой. */
export const curveStats = (rows: EpochRow[], key: string, lowerIsBetter: boolean) => {
  let best: { value: number; epoch: number } | null = null;
  let final: { value: number; epoch: number } | null = null;
  for (const row of rows) {
    const value = row[key];
    if (value === undefined) continue;
    final = { value, epoch: row.epoch };
    if (!best || (lowerIsBetter ? value < best.value : value > best.value)) {
      best = { value, epoch: row.epoch };
    }
  }
  return { best, final };
};
