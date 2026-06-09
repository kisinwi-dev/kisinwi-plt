// Сервис для запуска пайплайнов агентов (сервис agents, порт 6400).
import { handleResponse, serviceUrl } from './http';

// Базовый URL сервиса агентов берётся из переменной окружения VITE_AGENTS,
// если её нет – localhost:6400.
const AGENTS_URL = serviceUrl(import.meta.env.VITE_AGENTS, 'localhost:6400');

/** Параметры запуска пайплайна development. */
export interface StartDevelopmentPayload {
  dataset_id: string;
  version_id: string;
  model_name: string;
  deployment_constraints: string;
  business_requirements: string;
  denied_hypotheses_info?: string[];
  max_iter?: number;
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
};
