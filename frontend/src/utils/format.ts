export const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

// Проверяет, содержит ли ISO-строка обозначение часового пояса (Z или ±hh:mm).
const hasTimezone = (value: string): boolean => /[zZ]$|[+-]\d{2}:?\d{2}$/.test(value);

/**
 * Преобразует ISO-время от бэкенда в строку в часовом поясе пользователя.
 * Бэкенд отдаёт наивное UTC-время без обозначения зоны (datetime.now() в UTC-контейнере),
 * поэтому при отсутствии таймзоны строка трактуется как UTC.
 */
export const formatDateTime = (value: string | null | undefined): string => {
  if (!value) return '—';
  const normalized = hasTimezone(value) ? value : `${value}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
};

// Русская плюрализация: pluralRu(21, ['версия', 'версии', 'версий']) -> 'версия'.
export const pluralRu = (n: number, [one, few, many]: [string, string, string]): string => {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
};

// Значение метрики обучения: целые — как есть, дробные — 4 знака.
export const formatMetricValue = (v: number): string =>
  Number.isInteger(v) ? String(v) : v.toFixed(4);

// Форматирование длительности в мс/с (>= 1 с показываем в секундах).
export const formatDuration = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)} с` : `${ms.toFixed(0)} мс`;

// CSS-класс статуса для бейджа: 'IN PROGRESS' -> 'status-in-progress'.
export const statusClass = (status: string): string =>
  `status-${status.toLowerCase().replace(/\s+/g, '-')}`;

export const formatDateParts = (value: string | null | undefined): { date: string; time: string } => {
  if (!value) return { date: '—', time: '' };
  const normalized = hasTimezone(value) ? value : `${value}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return { date: value, time: '' };
  return {
    date: date.toLocaleDateString(),
    time: date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }),
  };
};