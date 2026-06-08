// Импортируем типы для моделей, файлов и параметров запроса.
import type {
  MLModel,
  MLModels,
  MLModelFiles,
  MLModelStatus,
  ModelsQuery,
} from '../types/mlModels';
import { handleResponse, serviceUrl } from './http';

// Базовый URL сервиса ml_models берётся из VITE_ML_MODELS, по умолчанию localhost:6300.
const ML_MODELS_URL = serviceUrl(import.meta.env.VITE_ML_MODELS, 'localhost:6300');

/**
 * Сервис для просмотра моделей и скачивания файлов весов из ml_models.
 */
export const mlModelsService = {
  /**
   * Получить список моделей с фильтрами и пагинацией.
   * GET /models?dataset_id&status&name&limit&offset
   */
  async getModels(query: ModelsQuery = {}): Promise<MLModels> {
    const url = new URL(`${ML_MODELS_URL}/models`);
    if (query.dataset_id) url.searchParams.append('dataset_id', query.dataset_id);
    if (query.status) url.searchParams.append('status', query.status);
    if (query.name) url.searchParams.append('name', query.name);
    if (query.limit != null) url.searchParams.append('limit', String(query.limit));
    if (query.offset != null) url.searchParams.append('offset', String(query.offset));
    const response = await fetch(url.toString());
    return handleResponse<MLModels>(response);
  },

  /**
   * Получить одну модель по ID.
   * GET /models/{modelId}
   */
  async getModel(modelId: string): Promise<MLModel> {
    const response = await fetch(`${ML_MODELS_URL}/models/${modelId}`);
    return handleResponse<MLModel>(response);
  },

  /**
   * Получить справочник статусов моделей (для фильтра).
   * GET /info/models/status
   */
  async getModelStatuses(): Promise<MLModelStatus[]> {
    const response = await fetch(`${ML_MODELS_URL}/info/models/status`);
    const data = await handleResponse<{ statuses: MLModelStatus[] }>(response);
    // На случай пустого ответа handleResponse вернёт true — нормализуем.
    return typeof data === 'object' && data?.statuses ? data.statuses : [];
  },

  /**
   * Получить список файлов весов модели. 204 → пустой список.
   * GET /models/{modelId}/files
   */
  async getModelFiles(modelId: string): Promise<MLModelFiles> {
    const response = await fetch(`${ML_MODELS_URL}/models/${modelId}/files`);
    const data = await handleResponse<MLModelFiles>(response);
    return typeof data === 'object' && data?.files ? data : { files: [] };
  },

  /**
   * Скачать файл весов: получаем blob и инициируем загрузку в браузере.
   * GET /models/files/{fileId}/download
   */
  async downloadFile(fileId: string, filename: string): Promise<void> {
    const response = await fetch(`${ML_MODELS_URL}/models/files/${fileId}/download`);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    const blob = await response.blob();
    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(objectUrl);
  },
};
