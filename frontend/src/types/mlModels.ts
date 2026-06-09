// Типы для сервиса ml_models (реестр обученных моделей и файлов весов).

// Одна модель из реестра ml_models.
export interface MLModel {
  id: string;
  name: string;
  version: number;
  model_type: string;
  status: string;
  description: string | null;
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

// Ответ списка моделей с метаданными пагинации.
export interface MLModels {
  models: MLModel[];
  total: number;
  limit: number | null;
  offset: number;
}

// Файл весов модели.
export interface MLModelFile {
  id: string;
  model_id: string;
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

// Группа версий одной модели (для grouped view).
export interface MLModelGroup {
  name: string;
  versions: MLModel[];  // отсортированы version DESC
}

// Ответ сгруппированного списка моделей.
export interface MLModelsGrouped {
  groups: MLModelGroup[];
  total: number;
  limit: number | null;
  offset: number;
}

// Параметры запроса grouped endpoint: name — частичный поиск по имени группы.
export type GroupedModelsQuery = ModelsQuery;
