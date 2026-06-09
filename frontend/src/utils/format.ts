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