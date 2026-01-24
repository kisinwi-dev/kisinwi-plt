# Dataset Storage Structure

[Русский](##русский) | [English](##english)

## Русский <a name="русский"></a>

В этом каталоге находятся все наборы данных, используемые системой.

### Общая структура

```
datasets/
    <dataset_type>/
        <dataset_task>/
            <dataset_name>/
                <version>/
                    <data>
```

### Описание уровней

1. dataset_type
   - изображение
   - текст
   - таблица

2. dataset_task
    Зависит от типа набора данных:
   - изображение: классификация, детекция, сегментация
   - текст: классификация, тональность
   - таблица: классификация, регрессия

3. dataset_name
    Уникальный идентификатор набора данных.

4. Версия
    Версия набора данных (`v_0`, `v_1`, ...)

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
- Проверка выполняется перед перемещением данных в этот каталог.


## English <a name="english"></a>

This directory contains all datasets used by the system.

### General Structure

```
datasets/
    <dataset_type>/
        <dataset_task>/
            <dataset_name>/
                <version>/
                    <data>
```

### Levels description

1. dataset_type
   - image
   - text
   - tabular

2. dataset_task
   Depends on dataset type:
   - image: classification, detection, segmentation
   - text: classification, sentiment, ner
   - tabular: classification, regression

3. dataset_name
   Unique dataset identifier.

4. version
   Dataset version (`v_0`, `v_1`, ...)

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

- Validation is performed before moving data into this directory.
