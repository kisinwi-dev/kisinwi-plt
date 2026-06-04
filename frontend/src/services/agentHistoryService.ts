// Импортируем типы для дискуссий, ответов агентов, системных сообщений и инструментов.
import type {
  AgentResponse,
  DiscussionListParams,
  DiscussionMeta,
  SystemMessage,
  Tool,
} from '../types/agentHistory';

// Базовый URL сервиса истории агентов берётся из переменной окружения
// VITE_AGENT_HISTORY, если её нет – localhost:6410.
const AGENT_HISTORY_URL = `http://${import.meta.env.VITE_AGENT_HISTORY ?? 'localhost:6410'}`;

/**
 * Универсальная функция обработки HTTP-ответа.
 * @param response – объект Response от fetch.
 * @returns промис с данными типа T.
 * @throws ошибка с текстом, извлечённым из тела ответа или статусом.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  // Если статус ответа не в диапазоне 200-299 – ошибка.
  if (!response.ok) {
    let errorMsg = `HTTP error ${response.status}`;
    try {
      // FastAPI отдаёт ошибки в формате { detail: ... }, поддержим и { message: ... }.
      const errorData = await response.json();
      errorMsg = errorData.detail ?? errorData.message ?? errorMsg;
    } catch {
      // Если не удалось распарсить JSON – оставляем стандартное сообщение.
    }
    throw new Error(errorMsg);
  }

  // Для пустых ответов (например, 204 No Content) читаем текст, если он пуст – возвращаем true.
  const text = await response.text();
  if (!text) return true as T;

  // Иначе парсим JSON и возвращаем результат.
  return JSON.parse(text);
}

/**
 * Сервис для чтения истории агентов: дискуссии, ответы, системные сообщения, инструменты.
 */
export const agentHistoryService = {
  /**
   * Получить список дискуссий с метаданными (с фильтрацией и пагинацией).
   * GET /discussions?status=&pipeline=&skip=&limit=
   */
  async getDiscussions(params: DiscussionListParams = {}): Promise<DiscussionMeta[]> {
    const url = new URL(`${AGENT_HISTORY_URL}/discussions`);
    if (params.status) url.searchParams.append('status', params.status);
    if (params.pipeline) url.searchParams.append('pipeline', params.pipeline);
    if (params.skip !== undefined) url.searchParams.append('skip', String(params.skip));
    if (params.limit !== undefined) url.searchParams.append('limit', String(params.limit));
    const response = await fetch(url.toString());
    return handleResponse<DiscussionMeta[]>(response);
  },

  /**
   * Получить метаданные одной дискуссии.
   * GET /discussions/{discussionId}/meta
   */
  async getDiscussionMeta(discussionId: string): Promise<DiscussionMeta> {
    const response = await fetch(`${AGENT_HISTORY_URL}/discussions/${discussionId}/meta`);
    return handleResponse<DiscussionMeta>(response);
  },

  /**
   * Получить все ответы агентов в рамках дискуссии (отсортированы по времени).
   * GET /discussions/{discussionId}/responses
   */
  async getResponses(discussionId: string): Promise<AgentResponse[]> {
    const response = await fetch(`${AGENT_HISTORY_URL}/discussions/${discussionId}/responses`);
    return handleResponse<AgentResponse[]>(response);
  },

  /**
   * Получить все системные сообщения дискуссии (отсортированы по времени).
   * GET /discussions/{discussionId}/system_messages
   */
  async getSystemMessages(discussionId: string): Promise<SystemMessage[]> {
    const response = await fetch(`${AGENT_HISTORY_URL}/discussions/${discussionId}/system_messages`);
    return handleResponse<SystemMessage[]>(response);
  },

  /**
   * Получить инструменты, вызванные в рамках конкретного запуска агента.
   * GET /discussions/{discussionId}/responses/{responseId}/tools
   */
  async getToolsByResponse(discussionId: string, responseId: string): Promise<Tool[]> {
    const response = await fetch(
      `${AGENT_HISTORY_URL}/discussions/${discussionId}/responses/${responseId}/tools`,
    );
    return handleResponse<Tool[]>(response);
  },
};
