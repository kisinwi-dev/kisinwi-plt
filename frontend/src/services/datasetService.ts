// Импортируем типы для датасетов, новых датасетов, новых версий и существующих версий.
import type { Dataset, NewDataset, NewVersion, Version } from '../types/dataset';

// Базовый URL API берётся из переменной окружения VITE_API_URL, если её нет – localhost:8000/api.
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

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
      // Пытаемся получить JSON с описанием ошибки (бэкенд может отдавать { message: ... })
      const errorData = await response.json();
      errorMsg = errorData.message || errorMsg;
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
 * Сервис для работы с датасетами и версиями.
 * Содержит методы для всех CRUD операций, а также загрузки файлов.
 */
export const datasetService = {
  // ==================== Датасеты ====================

  /**
   * Получить все датасеты с полными метаданными.
   * GET /api/datasets/
   */
  async getDatasets(): Promise<Dataset[]> {
    const response = await fetch(`${API_BASE_URL}/datasets/`);
    return handleResponse<Dataset[]>(response);
  },

  /**
   * Получить метаданные конкретного датасета по его ID.
   * @param datasetId – идентификатор датасета.
   * GET /api/datasets/{datasetId}
   */
  async getDataset(datasetId: string): Promise<Dataset> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`);
    return handleResponse<Dataset>(response);
  },

  /**
   * Создать новый датасет (только метаданные).
   * @param data – объект NewDataset (без файла).
   * POST /api/datasets/new
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
   * Удалить датасет по ID.
   * @param datasetId – идентификатор датасета.
   * DELETE /api/datasets/{datasetId}
   */
  async deleteDataset(datasetId: string): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`, {
      method: 'DELETE',
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Установить версию по умолчанию для датасета.
   * @param datasetId – идентификатор датасета.
   * @param versionId – идентификатор версии.
   * POST /api/datasets/{datasetId}/default_version?default_version={versionId}
   */
  async setDefaultVersion(datasetId: string, versionId: string): Promise<boolean> {
    const url = new URL(`${API_BASE_URL}/datasets/${datasetId}/default_version`);
    url.searchParams.append('default_version', versionId);
    const response = await fetch(url.toString(), { method: 'POST' });
    return handleResponse<boolean>(response);
  },

  // ==================== Версии ====================

  /**
   * Получить список версий для конкретного датасета.
   * @param datasetId – идентификатор датасета.
   * GET /api/datasets/{datasetId}/versions/
   */
  async getVersions(datasetId: string): Promise<Version[]> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/versions/`);
    return handleResponse<Version[]>(response);
  },

  /**
   * Создать новую версию (только метаданные).
   * @param datasetId – идентификатор датасета.
   * @param versionData – объект NewVersion.
   * POST /api/datasets/{datasetId}/versions/new
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
   * Удалить версию.
   * @param datasetId – идентификатор датасета.
   * @param versionId – идентификатор версии.
   * DELETE /api/datasets/{datasetId}/versions/{versionId}
   */
  async deleteVersion(datasetId: string, versionId: string): Promise<boolean> {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/versions/${versionId}`, {
      method: 'DELETE',
    });
    return handleResponse<boolean>(response);
  },

  // ==================== Загрузка файлов ====================

  /**
   * Загрузить файл для последующего связывания с датасетом или версией.
   * @param idData – идентификатор, под которым файл будет сохранён (dataset_id или version_id).
   * @param file – файл для загрузки.
   * POST /api/upload (multipart/form-data)
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