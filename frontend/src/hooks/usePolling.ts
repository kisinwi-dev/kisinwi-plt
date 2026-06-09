import { useEffect, useRef, useState, useCallback } from 'react';
import type { DependencyList } from 'react';

interface UsePollingOptions<T> {
  // Интервал между опросами, мс.
  intervalMs: number;
  // Запускать ли цикл (false — опрос остановлен). По умолчанию true.
  enabled?: boolean;
  // Делать первый запрос сразу при старте/перезапуске. По умолчанию true.
  immediate?: boolean;
  // Продолжать опрос только пока возвращает true. Если не задан — опрашиваем всегда.
  continueWhile?: (data: T) => boolean;
  // Обработчик ошибки запроса (например, показать уведомление).
  onError?: (err: unknown) => void;
  // При изменении любого значения цикл перезапускается (как deps у useEffect).
  deps?: DependencyList;
}

interface UsePollingResult<T> {
  data: T | undefined;
  loading: boolean;
  error: unknown;
  refetch: () => void;
}

/**
 * Опрос по интервалу через рекурсивный setTimeout (без наложения запросов).
 * Покрывает и setInterval-кейс (без continueWhile), и «опрашивать пока активно»
 * (continueWhile / enabled). fetcher вызывается через ref — вызывающему не нужна
 * мемоизация функции; перезапуск цикла контролируется через deps.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  options: UsePollingOptions<T>,
): UsePollingResult<T> {
  const { intervalMs, enabled = true, immediate = true, continueWhile, onError, deps = [] } = options;

  const [data, setData] = useState<T | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(undefined);

  // Актуальные ссылки на колбэки — чтобы не перезапускать эффект при их пересоздании.
  const fetcherRef = useRef(fetcher);
  const continueRef = useRef(continueWhile);
  const onErrorRef = useRef(onError);

  // Обновляем ссылки после каждого рендера (мутировать ref во время рендера нельзя).
  useEffect(() => {
    fetcherRef.current = fetcher;
    continueRef.current = continueWhile;
    onErrorRef.current = onError;
  });

  // Тик принудительного перезапуска для refetch().
  const [refetchTick, setRefetchTick] = useState(0);
  const refetch = useCallback(() => setRefetchTick(t => t + 1), []);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let firstLoad = true;

    const poll = async () => {
      try {
        if (firstLoad) setLoading(true);
        const result = await fetcherRef.current();
        if (cancelled) return;
        setData(result);
        setError(undefined);
        if (continueRef.current?.(result) ?? true) {
          timer = setTimeout(poll, intervalMs);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err);
        onErrorRef.current?.(err);
      } finally {
        if (!cancelled) setLoading(false);
        firstLoad = false;
      }
    };

    if (immediate) {
      poll();
    } else {
      timer = setTimeout(poll, intervalMs);
    }

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, enabled, immediate, refetchTick, ...deps]);

  return { data, loading, error, refetch };
}
