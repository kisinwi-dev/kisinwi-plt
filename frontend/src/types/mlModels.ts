// Типы для сервиса ml_models (реестр моделей: модель → версии, файлы весов).

// Версия модели (name/description денормализованы из родительской модели).
export interface MLModelVersion {
  id: string;
  model_id: string;
  name: string;
  description: string | null;
  version: number;
  model_type: string;
  status: string;
  metrics_report: string;
  classes: string[];
  // Конфиг обучения динамический — структуру не хардкодим.
  train_params: Record<string, unknown>;
  created_at: string;
  dataset_id: string;
  dataset_version_id: string;
  framework: string | null;
  framework_version: string | null;
}

// Модель (родитель) со списком версий (version DESC).
export interface MLModel {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  versions: MLModelVersion[];
}

// Ответ списка моделей с метаданными пагинации (по моделям).
export interface MLModels {
  models: MLModel[];
  total: number;
  limit: number | null;
  offset: number;
}

// Ответ плоского списка версий с метаданными пагинации.
export interface MLModelVersions {
  versions: MLModelVersion[];
  total: number;
  limit: number | null;
  offset: number;
}

// Файл весов версии модели.
export interface MLModelFile {
  id: string;
  version_id: string;
  filename: string;
  file_size: number;
  created_at: string;
}

export interface MLModelFiles {
  files: MLModelFile[];
}

// Справочник статусов моделей (GET /info/models/status).
export interface MLModelStatus {
  id: number;
  status: string;
  description: string;
}

// Параметры фильтрации/пагинации списка моделей.
export interface ModelsQuery {
  dataset_id?: string;
  status?: string;
  name?: string;
  limit?: number;
  offset?: number;
}

// Параметры плоского списка версий: дополнительно фильтр по родительской модели.
export interface VersionsQuery extends ModelsQuery {
  model_id?: string;
}
