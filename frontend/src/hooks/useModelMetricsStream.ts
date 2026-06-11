import { useEffect, useState } from 'react';
import { metricsService } from '../services/metricsService';
import type { ModelMetrics, TrainingStatus } from '../services/metricsService';

const FINAL_STATUSES: TrainingStatus[] = ['completed', 'failed', 'cancelled'];

interface UseModelMetricsStreamResult {
  data: ModelMetrics | undefined;
  status: TrainingStatus | null | undefined;
  finished: boolean;
  loading: boolean;
  error: boolean;
}

/**
 * Метрики модели в реальном времени через SSE: полный снимок приходит при
 * подключении и после каждой эпохи. Соединение не закрываем при ошибке —
 * EventSource переподключается сам, а очередной снимок восстанавливает данные.
 * По событию 'end' (финальный статус обучения) соединение закрываем сами:
 * сервер поток не завершает, новых снимков уже не будет.
 * Если SSE упал до первого снимка, делаем одноразовый fallback-запрос GET.
 */
export function useModelMetricsStream(modelId: string): UseModelMetricsStreamResult {
  const [data, setData] = useState<ModelMetrics | undefined>(undefined);
  const [finished, setFinished] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    setData(undefined);
    setFinished(false);
    setError(false);

    let cancelled = false;
    let gotData = false;
    let fallbackTried = false;

    const applyMetrics = (metrics: ModelMetrics) => {
      gotData = true;
      setData(metrics);
      setError(false);
      // Статус мог быть финальным уже в первом снимке (просмотр обученной модели).
      if (metrics.status && FINAL_STATUSES.includes(metrics.status)) setFinished(true);
    };

    const unsubscribe = metricsService.subscribeModelMetrics(modelId, {
      onMetrics: (metrics) => {
        if (cancelled) return;
        applyMetrics(metrics);
      },
      onEnd: () => {
        if (cancelled) return;
        setFinished(true);
        unsubscribe();
      },
      onError: () => {
        if (cancelled || gotData) return;
        setError(true);
        if (fallbackTried) return;
        fallbackTried = true;
        metricsService
          .getModelMetrics(modelId)
          .then((metrics) => {
            if (cancelled || gotData || metrics === null) return;
            applyMetrics(metrics);
          })
          .catch(() => undefined);
      },
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [modelId]);

  return {
    data,
    status: data?.status,
    finished,
    loading: data === undefined && !error,
    error,
  };
}
