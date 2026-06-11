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

export interface ModelMetrics {
  model_id: string;
  status?: TrainingStatus | null;
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
