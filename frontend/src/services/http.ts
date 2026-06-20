// Общие хелперы для работы с HTTP во всех сервисах.

/**
 * Базовый URL сервиса из переменной окружения Vite с фолбэком.
 * @param envValue значение import.meta.env.VITE_* (может быть undefined)
 * @param fallback host:port по умолчанию, напр. 'localhost:6300'
 */
export const serviceUrl = (envValue: string | undefined, fallback: string): string =>
  `http://${envValue ?? fallback}`;

/**
 * Собирает URL с query-параметрами, пропуская пустые (null/undefined/'').
 * @param base базовый URL без query, напр. `${ML_MODELS_URL}/models`
 * @param params пары имя→значение; пустые значения не добавляются
 */
export const buildUrl = (
  base: string,
  params: Record<string, string | number | undefined | null>,
): string => {
  const url = new URL(base);
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== '') url.searchParams.append(k, String(v));
  }
  return url.toString();
};

/**
 * Универсальная обработка HTTP-ответа.
 * @returns данные типа T; для пустых ответов (например, 204) — true.
 * @throws Error с текстом из тела ответа (FastAPI: detail/message) или статусом.
 */
export async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMsg = `HTTP error ${response.status}`;
    try {
      const errorData = await response.json();
      errorMsg = errorData.detail ?? errorData.message ?? errorMsg;
    } catch {
      // Тело не JSON — оставляем стандартное сообщение.
    }
    throw new Error(errorMsg);
  }

  const text = await response.text();
  if (!text) return true as T;
  return JSON.parse(text);
}
