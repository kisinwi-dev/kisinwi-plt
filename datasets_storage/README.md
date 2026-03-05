# Datasets_storage

В этом каталоге находятся все наборы данных, используемые системой.

## Общая структура

```
datasets/
├── <dataset_id>/                      # Уникальный идентификатор датасета
│   ├── metadata_ds.json               # Метаданные всего датасета
│   ├── v_1/                           # Первая версия (формируется при первой загрузке датасета)
│   │   ├── metadata_v.json            # Метаданные конкретной версии
│   │   ├── train/                     # Обучающая выборка
│   │   ├── val/                       # Валидационная выборка
│   │   └── test/                      # Тестовая выборка (опционально)
│   ├── v_2/                           # Следующая версия
│   │   ├── metadata_v.json
│   │   └── …
│   └── v_3/
│       └── …                          # … и так далее
└── another_dataset_id/                # Другой датасет
    ├── metadata_ds.json
    └── v_1/
        ├── metadata_v.json
        └── …
```

  
Пример заполненного файла **metadata_ds.json**:
```json
{
    "dataset_id": "cats_vs_dogs",
    "name": "Кошки и собаки",
    "description": "Бинарная классификация кошек и собак",
    "num_classes": 2,
    "class_names": ["cat", "dog"],
    "class_to_idx": {"cat": 0, "dog": 1},
    "source": {
        "kaggle_cat_vs_dog": {
            "url": "https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat",
            "description": "Фотографии кошек и собак 512х512 pix"
        }
    },
    "type": "image",
    "task": "classification",
    "created_at": "2026-03-01",
    "updated_at": "2026-03-05",
    "default_version_id": "v_1"
}
```

## Структура набора данных для классификации изображений

Пример сруктуры хранения файлов:

```
datasets/
└── cats_vs_dogs/                                 # dataset_id
    ├── metadata_ds.json                          # глобальное описание датасета
    ├── v_1/                                      # version_id = "v_1"
    │   ├── metadata_v.json                       # метаданные версии
    │   ├── train/                                # ~800 изображений
    │   │   ├── cat/                              # класс 0
    │   │   │   ├── 001_cat.jpg
    │   │   │   ├── 002_cat.png
    │   │   │   └── …
    │   │   └── dog/                              # класс 1
    │   │       ├── 001_dog_running.jpg
    │   │       └── …
    │   ├── val/
    │   │   ├── cat/
    │   │   └── dog/
    │   └── test/
    │       ├── cat/
    │       └── dog/
    │
    └── v_2-augmented/                            # другая версия (version_id = "v_2-augmented")
        ├── metadata_v.json
        ├── train/
        ├── val/
        └── test/
```

Пример заполненного файла **metadata_v.json**:

```json
{
    "version_id": "v_1",
    "parent_version_id": null,
    "name": "оригинальные фотографии",
    "description": "Версия с собранными данными из источников без преобработки",
    "created_at": "2026-03-05",
    "total_images": 1200,
    "splits": {
        "train": {"images": 800, "classes_distribution": {"cat": 400, "dog": 400}},
        "val":   {"images": 200, "classes_distribution": {"cat": 100, "dog": 100}},
        "test":  {"images": 200, "classes_distribution": {"cat": 100, "dog": 100}},
    },
    "resolution": {"width": 512, "height": 512},
    "augmentations_applied": false,
}
```

## Примечания
- Валидация данных происходит при загрузке данных автоматически.
