from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

class MetricesParams(BaseModel):
    metrics_list: List[str] = Field(
        default=['accuracy', 'precision', 'recall', 'f1'],
        description='Список метрик (loss добавляется автоматически)'
    )
    average: Optional[str] = Field(
        default='macro',
        description='Тип усреднения precision/recall/f1'
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "metrics_list": ['accuracy', 'precision', 'recall', 'f1'],
                "average": 'macro'
            }
        }
    }

class MetricesParamCollections(BaseModel):
    train_val: MetricesParams = Field(
        ...,
        description="Единый список метрик для train и val: одинаковый набор на обеих "
                    "выборках, чтобы каждую метрику можно было сравнивать между ними "
                    "(контроль переобучения)"
    )
    test: MetricesParams = Field(
        ...,
        description="Метрики для тестирования (могут расширять train/val-набор)"
    )

    @model_validator(mode='before')
    @classmethod
    def _merge_legacy_split_lists(cls, data):
        """Старый формат с раздельными train/val: объединяем списки в train_val."""
        if not isinstance(data, dict) or 'train_val' in data:
            return data
        legacy = [data.get(split) for split in ('train', 'val')]
        legacy = [params for params in legacy if isinstance(params, dict)]
        if not legacy:
            return data
        merged_list = list(dict.fromkeys(
            name for params in legacy for name in params.get('metrics_list') or []
        ))
        average = next(
            (params['average'] for params in legacy if params.get('average')),
            'macro'
        )
        train_val = {'average': average}
        if merged_list:  # пустой список не затирает дефолтный набор метрик
            train_val['metrics_list'] = merged_list
        data['train_val'] = train_val
        return data

    model_config = {
        "json_schema_extra": {
            "example": {
                "train_val": {
                    "metrics_list": ['accuracy', 'precision', 'recall', 'f1'],
                    "average": 'macro'
                },
                "test": {
                    "metrics_list": ['accuracy', 'precision', 'recall', 'f1', 'specificity'],
                    "average": 'macro'
                }
            }
        }
    }
