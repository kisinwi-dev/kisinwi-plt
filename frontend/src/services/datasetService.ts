// Импортируем типы для датасетов, новых датасетов, новых версий и существующих версий.
import type { Dataset, NewDataset, NewVersion, Version, VersionSplitsResponse } from '../types/dataset';
import type { FilesDiffResponse, VersionComparisonResponse } from '../types/datasetComparison';
import { handleResponse, serviceUrl, buildUrl } from './http';

// Базовый URL сервиса datasets берётся из переменной окружения VITE_DMS, по умолчанию localhost:6500.
const DMS_URL = serviceUrl(import.meta.env.VITE_DMS, 'localhost:6500');

/**
 * Сервис для работы с датасетами и версиями.
 * Содержит методы для всех CRUD операций, а также загрузки файлов.
 */
export const datasetService = {
  // ==================== Датасеты ====================

  /**
   * Получить все датасеты с полными метаданными.
   * GET /datasets/
   */
  async getDatasets(): Promise<Dataset[]> {
    const response = await fetch(`${DMS_URL}/datasets/`);
    return handleResponse<Dataset[]>(response);
  },

  /**
   * Получить метаданные конкретного датасета по его ID.
   * @param datasetId – идентификатор датасета.
   * GET /datasets/{datasetId}
   */
  async getDataset(datasetId: string): Promise<Dataset> {
    const response = await fetch(`${DMS_URL}/datasets/${datasetId}`);
    return handleResponse<Dataset>(response);
  },

  /**
   * Создать новый датасет (только метаданные).
   * @param data – объект NewDataset (без файла).
   * POST /datasets/new
   */
  async createDataset(data: NewDataset): Promise<boolean> {
    const response = await fetch(`${DMS_URL}/datasets/new`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Удалить датасет по ID.
   * @param datasetId – идентификатор датасета.
   * DELETE /datasets/{datasetId}
   */
  async deleteDataset(datasetId: string): Promise<boolean> {
    const response = await fetch(`${DMS_URL}/datasets/${datasetId}`, {
      method: 'DELETE',
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Установить версию по умолчанию для датасета.
   * @param datasetId – идентификатор датасета.
   * @param versionId – идентификатор версии.
   * POST /datasets/{datasetId}/default_version?default_version={versionId}
   */
  async setDefaultVersion(datasetId: string, versionId: string): Promise<boolean> {
    const url = buildUrl(`${DMS_URL}/datasets/${datasetId}/default_version`, {
      default_version: versionId,
    });
    const response = await fetch(url, { method: 'POST' });
    return handleResponse<boolean>(response);
  },

  // ==================== Версии ====================

  /**
   * Получить список версий для конкретного датасета.
   * @param datasetId – идентификатор датасета.
   * GET /datasets/{datasetId}/versions/
   */
  async getVersions(datasetId: string): Promise<Version[]> {
    const response = await fetch(`${DMS_URL}/datasets/${datasetId}/versions/`);
    return handleResponse<Version[]>(response);
  },

  /**
   * Создать новую версию (только метаданные).
   * @param datasetId – идентификатор датасета.
   * @param versionData – объект NewVersion.
   * POST /datasets/{datasetId}/versions/new
   */
  async createVersion(datasetId: string, versionData: NewVersion): Promise<boolean> {
    const response = await fetch(`${DMS_URL}/datasets/${datasetId}/versions/new`, {
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
   * DELETE /datasets/{datasetId}/versions/{versionId}
   */
  async deleteVersion(datasetId: string, versionId: string): Promise<boolean> {
    const response = await fetch(`${DMS_URL}/datasets/${datasetId}/versions/${versionId}`, {
      method: 'DELETE',
    });
    return handleResponse<boolean>(response);
  },

  /**
   * Получить статистику по сплитам версии.
   * GET /datasets/{datasetId}/versions/{versionId}/splits
   */
  async getVersionSplits(datasetId: string, versionId: string): Promise<VersionSplitsResponse> {
    const response = await fetch(`${DMS_URL}/datasets/${datasetId}/versions/${versionId}/splits`);
    return handleResponse<VersionSplitsResponse>(response);
  },

  /**
   * Получить полную сводку сравнения двух версий датасета.
   * @param datasetId – идентификатор датасета.
   * @param fromVersionId – идентификатор базовой версии.
   * @param toVersionId – идентификатор сравниваемой версии.
   * GET /datasets/{datasetId}/versions/compare?from={fromVersionId}&to={toVersionId}
   */
  async compareVersions(
    datasetId: string,
    fromVersionId: string,
    toVersionId: string,
  ): Promise<VersionComparisonResponse> {
    const url = new URL(`${DMS_URL}/datasets/${datasetId}/versions/compare`);
    url.searchParams.append('from', fromVersionId);
    url.searchParams.append('to', toVersionId);
    const response = await fetch(url.toString());
    return handleResponse<VersionComparisonResponse>(response);
  },

  /**
   * Получить по-файловый diff двух версий (списки добавленных/удалённых файлов).
   * @param datasetId – идентификатор датасета.
   * @param fromVersionId – идентификатор базовой версии.
   * @param toVersionId – идентификатор сравниваемой версии.
   * GET /datasets/{datasetId}/versions/compare/files?from={fromVersionId}&to={toVersionId}
   */
  async compareVersionFiles(
    datasetId: string,
    fromVersionId: string,
    toVersionId: string,
  ): Promise<FilesDiffResponse> {
    const url = new URL(`${DMS_URL}/datasets/${datasetId}/versions/compare/files`);
    url.searchParams.append('from', fromVersionId);
    url.searchParams.append('to', toVersionId);
    const response = await fetch(url.toString());
    return handleResponse<FilesDiffResponse>(response);
  },

  // ==================== Загрузка файлов ====================

  /**
   * Загрузить файл для последующего связывания с датасетом или версией.
   * @param idData – идентификатор, под которым файл будет сохранён (dataset_id или version_id).
   * @param file – файл для загрузки.
   * POST /upload (multipart/form-data)
   */
  async uploadFile(idData: string, file: File): Promise<boolean> {
    const formData = new FormData();
    formData.append('id_data', idData);
    formData.append('file', file);
    const response = await fetch(`${DMS_URL}/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<boolean>(response);
  },
};