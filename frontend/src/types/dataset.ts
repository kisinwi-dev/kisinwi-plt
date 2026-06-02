export interface SourceItem {
  type: 'kaggle' | 'url' | 'huggingface' | 'other';
  url: string | null;
  description: string;
}
export type Source = SourceItem[];

export interface Version {
  id: string;
  name: string;
  description: string;
  sources: Source;
  num_samples: number;
  size_bytes: number;
  image_format_stats: Record<string, number>;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  classes_count: number;
  classes_names: string[];
  classes_to_idx: Record<string, number>;
  type: 'image' | 'text' | 'tabular' | 'other';
  task: 'classification' | 'regression' | 'detection' | 'segmentation' | 'other';
  default_version_id: string;
  versions: Version[];
  created_at: string;
  updated_at: string;
}

export interface NewVersion {
  id_data: string;
  name: string;
  description: string;
  sources: Source;
}

export interface NewDataset {
  name: string;
  description: string;
  type: 'image' | 'text' | 'tabular' | 'other';
  task: 'classification' | 'regression' | 'detection' | 'segmentation' | 'other';
  version: NewVersion;
}
