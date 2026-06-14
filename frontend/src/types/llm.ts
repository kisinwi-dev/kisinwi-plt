// Типы настройки LLM-модели агентов (сервис agents, /settings/llm).

/** Описание одной модели из каталога. */
export interface LlmModelInfo {
  /** OpenRouter id модели. */
  id: string;
  /** Человекочитаемое имя для UI. */
  label: string;
  /** Принимает ли модель кастомный temperature. */
  supports_temperature: boolean;
  /** Заметки и ограничения по модели. */
  notes: string;
}

/** Текущее состояние настройки модели агентов. */
export interface LlmSettings {
  /** Сейчас выбранная модель агентов. */
  current_model: string;
  /** Модель по умолчанию (из env). */
  default_model: string;
  /** Текущая модель вне каталога (кастомная). */
  is_custom: boolean;
  /** Каталог доступных моделей. */
  available: LlmModelInfo[];
}
