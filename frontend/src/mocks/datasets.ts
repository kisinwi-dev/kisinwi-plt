import type { Dataset } from '../types/dataset';

export const mockDatasets: Dataset[] = [
  {
    dataset_id: 'cifar10',
    name: 'CIFAR-10',
    description: 'Dataset with 10 classes of small images (32x32)',
    num_classes: 10,
    class_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    class_to_idx: {
      airplane: 0,
      automobile: 1,
      bird: 2,
      cat: 3,
      deer: 4,
      dog: 5,
      frog: 6,
      horse: 7,
      ship: 8,
      truck: 9,
    },
    sources: [
      {
        type: 'kaggle',
        url: 'https://www.kaggle.com/c/cifar-10',
        description: 'CIFAR-10 dataset from Kaggle',
      }
    ],
    type: 'image',
    task: 'classification',
    default_version_id: 'v1',
    versions: [
      {
        version_id: 'v1',
        description: 'Initial version',
        size_bytes: 163_000_000,
        num_samples: 60000,
        num_train: 50000,
        num_val: 0,
        num_test: 10000,
        created_at: '2025-01-15T10:00:00Z',
      },
      {
        version_id: 'v2',
        description: 'Augmented version',
        size_bytes: 175_000_000,
        num_samples: 60000,
        num_train: 50000,
        num_val: 0,
        num_test: 10000,
        created_at: '2025-02-10T12:30:00Z',
      },
    ],
    created_at: '2025-01-15T09:00:00Z',
    updated_at: '2025-02-10T12:30:00Z',
  },
  {
    dataset_id: 'flowers',
    name: 'Flowers Recognition',
    description: 'Photos of 5 types of flowers',
    num_classes: 5,
    class_names: ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    class_to_idx: {
      daisy: 0,
      dandelion: 1,
      rose: 2,
      sunflower: 3,
      tulip: 4,
    },
    sources: [
      {
        type: 'kaggle',
        url: 'https://www.kaggle.com/datasets/alxmamaev/flowers-recognition',
        description: 'Flowers dataset from Kaggle',
      }
    ],
    type: 'image',
    task: 'classification',
    default_version_id: 'v1',
    versions: [
      {
        version_id: 'v1',
        description: 'Original',
        size_bytes: 220_000_000,
        num_samples: 3670,
        num_train: 2936,
        num_val: 0,
        num_test: 734,
        created_at: '2025-02-20T14:15:00Z',
      },
    ],
    created_at: '2025-02-20T14:00:00Z',
    updated_at: '2025-02-20T14:15:00Z',
  },
];