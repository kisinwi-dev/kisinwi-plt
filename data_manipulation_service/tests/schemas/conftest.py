import pytest
import json
from pydantic import ValidationError
from typing import List, Optional
from pathlib import Path
from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset


@pytest.fixture()
def dsm_cd() -> DatasetMetadata:
    """"Экземпляр cats_vs_dogs класса DatasetMetadata"""
    return load_dsm('cats_vs_dogs')

@pytest.fixture()
def dsm_a() -> DatasetMetadata:
    """"Экземпляр apple класса DatasetMetadata"""
    return load_dsm('apple')

@pytest.fixture()
def ds_new() -> NewDataset:
    """Экземпляр apple класса NewDataset"""
    path = Path(__file__).parent.parent / 'fixture' / 'dataset_new.json'

    try:
        with path.open('r', encoding="utf-8") as f:
            data = json.load(f)
        return NewDataset.model_validate(data)
    except FileNotFoundError:
        raise ValueError(f"Файл не найден: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Невалидный JSON в файле {path}: {e}")
    except ValidationError as e:
        raise ValueError(f"Структура метаданных некорректна: {e}")

def load_dsm(dataset_id) -> DatasetMetadata:
    path = Path(__file__).parent.parent / 'fixture' / 'dataset' / dataset_id / 'metadata_ds.json'
    
    print(f"Загрузка файла: {path}")  # Для отладки
    
    try:
        with path.open('r', encoding="utf-8") as f:
            data = json.load(f)
        return DatasetMetadata.model_validate(data)
    except FileNotFoundError:
        raise ValueError(f"Файл не найден: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Невалидный JSON в файле {path}: {e}")
    except ValidationError as e:
        raise ValueError(f"Структура метаданных некорректна: {e}")