import { useEffect, useState } from 'react';

/**
 * Возвращает значение с задержкой: обновляется только после паузы во вводе.
 * Полезно, чтобы не слать запрос на каждое нажатие клавиши.
 */
export const useDebouncedValue = <T>(value: T, delay = 350): T => {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debounced;
};
