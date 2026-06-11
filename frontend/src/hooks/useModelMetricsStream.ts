import { useEffect, useState } from 'react';
import { metricsService } from '../services/metricsService';
import type { ModelMetrics } from '../services/metricsService';

interface UseModelMetricsStreamResult {
  data: ModelMetrics | undefined;
  loading: boolean;
  error: boolean;
}

/**
 * Метрики модели в реальном времени через SSE: полный снимок приходит при
 * подключении и после каждой эпохи. Соединение не закрываем при ошибке —
 * EventSource переподключается сам, а очередной снимок восстанавливает данные.
 * Если SSE упал до первого снимка, делаем одноразовый fallback-запрос GET.
 */
export function useModelMetricsStream(modelId: string): UseModelMetricsStreamResult {
  const [data, setData] = useState<ModelMetrics | undefined>(undefined);
  const [error, setError] = useState(false);

  useEffect(() => {
    setData(undefined);
    setError(false);

    let cancelled = false;
    let gotData = false;
    let fallbackTried = false;

    const unsubscribe = metricsService.subscribeModelMetrics(modelId, {
      onMetrics: (metrics) => {
        if (cancelled) return;
        gotData = true;
        setData(metrics);
        setError(false);
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
            gotData = true;
            setData(metrics);
            setError(false);
          })
          .catch(() => undefined);
      },
    });

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [modelId]);

  return { data, loading: data === undefined && !error, error };
}
