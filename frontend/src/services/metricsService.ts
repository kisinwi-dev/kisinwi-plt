import { serviceUrl, handleResponse } from './http';

const BASE = serviceUrl(import.meta.env.VITE_METRICS, 'localhost:6310');

export interface MetricData {
  name: string;
  values: number[];
}

export interface ModelMetrics {
  model_id: string;
  metrics: MetricData[];
}

export const metricsService = {
  async getModelMetrics(modelId: string): Promise<ModelMetrics | null> {
    const response = await fetch(`${BASE}/models/${modelId}`);
    if (response.status === 404) return null;
    return handleResponse<ModelMetrics>(response);
  },
};
