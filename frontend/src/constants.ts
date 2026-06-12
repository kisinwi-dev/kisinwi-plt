// Общие константы приложения: размеры страниц, лимиты UI, интервалы опроса.

// Сколько моделей грузим на одну страницу списка моделей.
export const MODELS_PAGE_SIZE = 12;

// Интервал опроса прогресса активной дискуссии (meta и лента сообщений).
export const POLL_INTERVAL_DISCUSSION_MS = 3000;

// Интервал опроса задачи обучения в tasker (виджет прогресса на странице модели).
export const POLL_INTERVAL_TASK_MS = 3000;

// Человекочитаемые статусы версии модели. Источник статуса — только реестр
// ml_models; справочник значений отдаёт GET /info/models/status.
export const MODEL_STATUS_LABELS: Record<string, string> = {
  completed: 'Обучена',
  training: 'Обучается',
  in_progress: 'Обучается',
  draft: 'Не обучена',
  failed: 'Не обучена',
  cancelled: 'Не обучена',
};

// Неизвестный статус показываем как есть, чтобы не скрывать новые значения из БД.
export const modelStatusLabel = (status: string): string =>
  MODEL_STATUS_LABELS[status] ?? status;

// CSS-класс бейджа статуса: классы в components.css — через дефис
// (status-in-progress), а статусы из БД — с подчёркиванием.
export const statusBadgeClass = (status: string): string =>
  `status-badge status-${status.replace(/_/g, '-')}`;
