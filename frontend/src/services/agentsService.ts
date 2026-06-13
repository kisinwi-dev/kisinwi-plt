// Сервис для запуска пайплайнов агентов (сервис agents, порт 6400).
import { handleResponse, serviceUrl } from './http';

// Базовый URL сервиса агентов берётся из переменной окружения VITE_AGENTS,
// если её нет – localhost:6400.
const AGENTS_URL = serviceUrl(import.meta.env.VITE_AGENTS, 'localhost:6400');

/** Параметры запуска пайплайна development. */
export interface StartDevelopmentPayload {
  dataset_id: string;
  version_id: string;
  /** Имя модели (обязательно, если не задан model_id). */
  model_name?: string;
  /** ID существующей модели — агенты обучат новые версии под ней. */
  model_id?: string;
  deployment_constraints: string;
  business_requirements: string;
  denied_hypotheses_info?: string[];
  max_iter?: number;
  title?: string;
  tags?: string[];
}

/** Параметры запуска быстрого пайплайна (ML-инженер + аналитик метрик). */
export interface StartQuickPayload {
  dataset_id: string;
  version_id: string;
  /** Имя модели (обязательно, если не задан model_id). */
  model_name?: string;
  /** ID существующей модели — агенты обучат новые версии под ней. */
  model_id?: string;
  deployment_constraints: string;
  business_requirements: string;
  title?: string;
  tags?: string[];
}

/** Ответ на запуск: id созданной дискуссии и её статус. */
export interface StartDevelopmentResult {
  discussion_id: string;
  status: string;
}

/**
 * Сервис запуска пайплайнов агентов.
 */
export const agentsService = {
  /**
   * Запустить полный пайплайн development асинхронно.
   * Дискуссия создаётся сразу, пайплайн выполняется в фоне.
   * POST /development/start
   */
  async startDevelopment(payload: StartDevelopmentPayload): Promise<StartDevelopmentResult> {
    const response = await fetch(`${AGENTS_URL}/development/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<StartDevelopmentResult>(response);
  },

  /**
   * Запустить быстрый пайплайн (ML-инженер + аналитик метрик) асинхронно.
   * Один проход без итераций: конфигурация, обучение, анализ метрик.
   * POST /quick/start
   */
  async startQuickTraining(payload: StartQuickPayload): Promise<StartDevelopmentResult> {
    const response = await fetch(`${AGENTS_URL}/quick/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<StartDevelopmentResult>(response);
  },
};
