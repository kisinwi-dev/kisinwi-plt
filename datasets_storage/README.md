# Dataset Storage Structure

[Русский](##русский) | [English](##english)

## Русский <a name="русский"></a>

В этом каталоге находятся все наборы данных, используемые системой.

### Общая структура

```
datasets/
    <dataset_name>/
        config.json
        <version>/
            config.json
            <data>
```

### Описание уровней

1. dataset_name
    Имя датасета.
    * config.json - информация о конкретном датасете 

2. Версия
    Версия набора данных (`v_0`, `v_1`, ...)
    * config.json - информация о версии датасета

### Структура набора данных для классификации изображений

```
<version>/
    train/
        <class_name>/
            *.jpg, *.png
    valid/
        <class_name>/
    test/
        <class_name>/
```

### Примечания
- Валидация данных происходит при загрузке данных автоматически.


## English <a name="english"></a>

This directory contains all datasets used by the system.

### General Structure

```
datasets/
    <dataset_name>/
        config.json
        <version>/
            config.json
            <data>
```

### Levels description

1. dataset_name
    The name of the dataset.
    * config.json — information about a specific dataset

2. Version
    The dataset version (v_0, v_1, ...)
   * config.json — information about the dataset version

### Image classification dataset structure

```
<version>/
    train/
        <class_name>/
            *.jpg, *.png
    valid/
        <class_name>/
    test/
        <class_name>/
```

### Notes

- Data validation is performed automatically during data upload.
