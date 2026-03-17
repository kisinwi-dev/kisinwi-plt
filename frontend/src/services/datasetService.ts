import type { Dataset, NewDataset, NewVersion, Version } from '../types/dataset';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

/**
 * Обрабатывает ответ от сервера.
 * - Если ответ успешный (ok), парсит JSON или возвращает true для пустых ответов.
 * - Если ошибка, пытается извлечь сообщение из тела ответа и выбрасывает ошибку.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMsg = `HTTP error ${response.status}`;
    try {
      const errorData = await response.json();
      errorMsg = errorData.message || errorMsg;
    } catch {
      // ignore
    }
    throw new Error(errorMsg);
  }

  // Для статусов 204 No Content или пустого тела
  const text = await response.text();
  if (!text) return true as T;

  return JSON.parse(text);
}

export const datasetService = {
  /**
   * Получить все датасеты (полные метаданные)
   */
  async getDatasets(): Promise<Dataset[]> {
    const response = await fetch(`${API_BASE_URL}/datasets/`);
    return handleResponse<Dataset[]>(response);
  },

  /**
   * Получить метаданные конкретного датасета по его ID
   */
  async getDataset(datasetId: string): Promise<Dataset> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`);
    return handleResponse<Dataset>(response);
  },

  /**
   * Создать новый датасет (только метаданные)
   */
  async createDataset(data: NewDataset): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/datasets/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Удалить датасет по ID
   */
  async deleteDataset(datasetId: string): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`, {
      method: 'DELETE',
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Установить версию по умолчанию для датасета
   */
  async setDefaultVersion(datasetId: string, versionId: string): Promise<boolean> {
    const url = new URL(`${API_BASE_URL}/datasets/${datasetId}/default_version`);
    url.searchParams.append('default_version', versionId);
    const response = await fetch(url.toString(), { method: 'POST' });
    return handleResponse<boolean>(response);
  },

  // --- Версии ---

  /**
   * Получить список версий для конкретного датасета
   */
  async getVersions(datasetId: string): Promise<Version[]> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/versions/`);
    return handleResponse<Version[]>(response);
  },

  /**
   * Создать новую версию (только метаданные)
   */
  async createVersion(datasetId: string, versionData: NewVersion): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/versions/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(versionData),
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Удалить версию
   */
  async deleteVersion(datasetId: string, versionId: string): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/versions/${versionId}`, {
      method: 'DELETE',
    });
    return handleResponse<boolean>(response);
  },

  // --- Загрузка файлов (общий эндпоинт) ---

  /**
   * Загрузить файл для последующего связывания с датасетом или версией.
   * @param idData – идентификатор, под которым файл будет сохранён (dataset_id или version_id)
   * @param file – файл для загрузки
   */
  async uploadFile(idData: string, file: File): Promise<boolean> {
    const formData = new FormData();
    formData.append('id_data', idData);
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<boolean>(response);
  },
};