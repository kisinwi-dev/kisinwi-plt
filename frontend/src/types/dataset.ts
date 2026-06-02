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

export interface SplitStats {
  total_samples: number;
  num_classes: number;
  balance_ratio: number;
  is_balanced: boolean;
  class_distribution: { class_name: string; count: number; percentage: number }[];
}

export interface ImageSizeStats {
  unique_sizes: number;
  total_images: number;
  most_common_size: string;
  most_common_count: number;
  size_consistency: number;
  top_10_sizes: Record<string, number>;
}

export interface VersionSplitsResponse {
  id: string;
  name: string;
  num_samples: number;
  size_bytes: number;
  overall_balance: number;
  splits_summary: Record<string, SplitStats>;
  image_size_stats: Record<string, ImageSizeStats>;
}

export interface NewDataset {
  name: string;
  description: string;
  type: 'image' | 'text' | 'tabular' | 'other';
  task: 'classification' | 'regression' | 'detection' | 'segmentation' | 'other';
  version: NewVersion;
}
