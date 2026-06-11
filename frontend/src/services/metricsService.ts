import { serviceUrl, handleResponse } from './http';

const BASE = serviceUrl(import.meta.env.VITE_METRICS, 'localhost:6310');

export type Split = 'train' | 'val' | 'test';

export interface MetricData {
  name: string;
  split?: Split;
  values: number[];
}

export interface ModelMetrics {
  model_id: string;
  train: MetricData[];
  val: MetricData[];
  test: MetricData[];
}

interface MetricsStreamHandlers {
  onMetrics: (metrics: ModelMetrics) => void;
  onError?: (event: Event) => void;
}

export const metricsService = {
  async getModelMetrics(modelId: string): Promise<ModelMetrics | null> {
    const response = await fetch(`${BASE}/models/${modelId}`);
    if (response.status === 404) return null;
    return handleResponse<ModelMetrics>(response);
  },

  /**
   * Подписка на SSE-поток метрик модели: сервер шлёт событие 'metrics'
   * с полным снимком при подключении и после каждой эпохи обучения.
   * При обрыве соединения EventSource переподключается сам.
   * @returns функция отписки (закрывает соединение)
   */
  subscribeModelMetrics(modelId: string, handlers: MetricsStreamHandlers): () => void {
    const source = new EventSource(`${BASE}/models/${modelId}/stream`);
    source.addEventListener('metrics', (event: MessageEvent) => {
      handlers.onMetrics(JSON.parse(event.data) as ModelMetrics);
    });
    source.onerror = (event) => handlers.onError?.(event);
    return () => source.close();
  },
};
