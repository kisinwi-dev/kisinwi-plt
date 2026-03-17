export interface SourceItem {
  type: 'kaggle' | 'url' | 'huggingface' | 'other';
  url: string;
  description: string;
}
export type Source = SourceItem[];

export interface Version {
  version_id: string;
  description: string;
  size_bytes: number;
  num_samples: number;
  num_train: number;
  num_val: number;
  num_test: number;
  created_at: string; // ISO date string
}

export interface Dataset {
  dataset_id: string;
  name: string;
  description: string;
  num_classes: number;
  class_names: string[];
  class_to_idx: Record<string, number>;
  sources: Source;
  type: 'image' | 'text' | 'tabular' | 'other';
  task: 'classification' | 'regression' | 'detection' | 'segmentation' | 'other';
  default_version_id: string;
  versions: Version[];
  created_at: string;
  updated_at: string;
}

// Новые типы для создания
export interface NewVersion {
  version_id: string;
  description: string;
}

export interface NewDataset {
  dataset_id: string;
  name: string;
  description: string;
  class_names: string[];
  sources: Source;
  type: 'image' | 'text' | 'tabular' | 'other';
  task: 'classification' | 'regression' | 'detection' | 'segmentation' | 'other';
  version: NewVersion;
}
