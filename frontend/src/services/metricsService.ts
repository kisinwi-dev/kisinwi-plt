import { serviceUrl, handleResponse } from './http';

const BASE = serviceUrl(import.meta.env.VITE_METRICS, 'localhost:6310');

export type Split = 'train' | 'val' | 'test';

export type TrainingStatus = 'in_progress' | 'completed' | 'failed' | 'cancelled';

export interface MetricData {
  name: string;
  split?: Split;
  values: number[];
  // Параллелен values (UTC); у старых моделей может быть короче.
  timestamps?: string[];
}

export interface CheckpointInfo {
  // Эпоха, веса которой сохранены (нумерация с 1).
  epoch: number;
  // Early-stop-метрика выбора лучшей эпохи (чистое имя, val-выборка).
  metric: string;
  // Значение метрики на эпохе чекпоинта; null — улучшение не фиксировалось,
  // сохранены веса финальной эпохи.
  value: number | null;
}

export interface ModelMetrics {
  model_id: string;
  status?: TrainingStatus | null;
  // null у моделей, обученных до ввода чекпоинтов.
  checkpoint?: CheckpointInfo | null;
  train: MetricData[];
  val: MetricData[];
  test: MetricData[];
}

export interface PerClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ClassReport {
  model_id: string;
  labels: string[];
  confusion_matrix: number[][];
  per_class: PerClassMetrics[];
}

export interface StreamEndEvent {
  model_id: string;
  status: TrainingStatus;
}

export interface CompareModelEntry {
  model_id: string;
  final_value: number;
  best_value: number;
  best_epoch: number;
  epochs: number;
  // Отставание от лидера по best_value (0 у лидера).
  delta_best: number;
}

export interface CompareMetric {
  metric: string;
  higher_is_better: boolean;
  best_model_id: string | null;
  models: CompareModelEntry[];
}

export interface ModelsCompareResponse {
  split: Split;
  metrics: CompareMetric[];
  // Модели без сохранённых метрик (не ошибка).
  missing: string[];
}

interface MetricsStreamHandlers {
  onMetrics: (metrics: ModelMetrics) => void;
  onEnd?: (event: StreamEndEvent) => void;
  onError?: (event: Event) => void;
}

export const metricsService = {
  async getModelMetrics(modelId: string): Promise<ModelMetrics | null> {
    const response = await fetch(`${BASE}/models/${modelId}`);
    if (response.status === 404) return null;
    return handleResponse<ModelMetrics>(response);
  },

  /** Сравнение моделей (мин. 2) по метрикам выборки; модели без метрик попадают в missing. */
  async compareModels(modelIds: string[], split: Split = 'val'): Promise<ModelsCompareResponse> {
    const response = await fetch(`${BASE}/models/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_ids: modelIds, split }),
    });
    return handleResponse<ModelsCompareResponse>(response);
  },

  /** Отчёт по классам на test; null — отчёта ещё нет (появляется после обучения). */
  async getClassReport(modelId: string): Promise<ClassReport | null> {
    const response = await fetch(`${BASE}/models/${modelId}/class-report`);
    if (response.status === 404) return null;
    return handleResponse<ClassReport>(response);
  },

  /**
   * Подписка на SSE-поток метрик модели: сервер шлёт событие 'metrics'
   * с полным снимком при подключении и после каждой эпохи обучения.
   * При финальном статусе обучения после снимка приходит событие 'end' —
   * сервер поток не закрывает, закрыть соединение должен подписчик.
   * При обрыве соединения EventSource переподключается сам.
   * @returns функция отписки (закрывает соединение)
   */
  subscribeModelMetrics(modelId: string, handlers: MetricsStreamHandlers): () => void {
    const source = new EventSource(`${BASE}/models/${modelId}/stream`);
    source.addEventListener('metrics', (event: MessageEvent) => {
      handlers.onMetrics(JSON.parse(event.data) as ModelMetrics);
    });
    source.addEventListener('end', (event: MessageEvent) => {
      handlers.onEnd?.(JSON.parse(event.data) as StreamEndEvent);
    });
    source.onerror = (event) => handlers.onError?.(event);
    return () => source.close();
  },
};
