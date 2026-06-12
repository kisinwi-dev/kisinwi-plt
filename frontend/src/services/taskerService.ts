import type { TrainingTasks, TasksQuery } from '../types/tasks';
import { handleResponse, serviceUrl } from './http';

// Базовый URL сервиса задач берётся из VITE_TASKER, по умолчанию localhost:6110.
const TASKER_URL = serviceUrl(import.meta.env.VITE_TASKER, 'localhost:6110');

/**
 * Сервис задач обучения: просмотр прогресса и отмена.
 */
export const taskerService = {
  /**
   * Получить задачи обучения с фильтрами.
   * GET /tasks?status&model_id
   */
  async getTasks(query: TasksQuery = {}): Promise<TrainingTasks> {
    const url = new URL(`${TASKER_URL}/tasks`);
    if (query.status) url.searchParams.append('status', query.status);
    if (query.model_id) url.searchParams.append('model_id', query.model_id);
    const response = await fetch(url.toString());
    return handleResponse<TrainingTasks>(response);
  },

  /**
   * Отменить задачу (только waiting/running); воркер остановит обучение
   * на границе эпохи.
   * POST /tasks/{taskId}/cancel
   */
  async cancelTask(taskId: string): Promise<void> {
    const response = await fetch(`${TASKER_URL}/tasks/${taskId}/cancel`, { method: 'POST' });
    await handleResponse<unknown>(response);
  },
};
