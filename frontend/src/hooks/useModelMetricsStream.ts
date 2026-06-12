import { useEffect, useState } from 'react';
import { metricsService } from '../services/metricsService';
import type { ModelMetrics, TrainingStatus } from '../services/metricsService';

const FINAL_STATUSES: TrainingStatus[] = ['completed', 'failed', 'cancelled'];

export interface UseModelMetricsStreamResult {
  data: ModelMetrics | undefined;
  /**
   * Стрим дошёл до финального статуса обучения. Это сигнал «новых снимков
   * не будет», а не статус модели — «обучена/не обучена» знает только
   * реестр ml_models.
   */
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

  // Сброс при смене модели — синхронно в рендере (паттерн «storing information
  // from previous renders» из доков React), чтобы между сменой id и подпиской
  // не отдавались данные прошлой модели.
  const [prevModelId, setPrevModelId] = useState(modelId);
  if (prevModelId !== modelId) {
    setPrevModelId(modelId);
    setData(undefined);
    setFinished(false);
    setError(false);
  }

  useEffect(() => {
    if (!modelId) return;

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
            if (cancelled || gotData) return;
            if (metrics === null) {
              // 404 — метрик у модели нет: это «пусто», а не ошибка сервиса.
              setData({ model_id: modelId, train: [], val: [], test: [] });
              setError(false);
              return;
            }
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
    finished,
    loading: data === undefined && !error,
    error,
  };
}
