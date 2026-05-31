from enum import Enum
from typing import Dict, List, Counter
from pydantic import BaseModel, Field

class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class ClassDistribution(BaseModel):
    """Распределение классов для одного сплита"""
    class_name: str = Field(..., description="Имя класса")
    class_id: int = Field(..., description="Id класса")
    count: int = Field(..., description="Количество обьектов")
    percentage: float = Field(..., description="Процент от общего количества в выборке")
    image_size_count: Dict[str, int] = Field(..., description="Количество изображений, с определённым размером")

    def to_summary(self) -> Dict:
        """Краткая информация о классе"""
        total_images = sum(self.image_size_count.values())
        unique_sizes = len(self.image_size_count)

        most_common_size = max(self.image_size_count, key=lambda size: self.image_size_count[size])
        most_common_count = self.image_size_count[most_common_size]

        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "count": self.count,
            "percentage": self.percentage,
            "size_stats": {
                "unique_sizes": unique_sizes,
                "most_common_size": most_common_size,
                "size_consistency": round(most_common_count / total_images, 2) if total_images > 0 else 0
            }
        }
    
    def get_balance_info(self) -> Dict:
        """Информация для расчёта баланса"""
        return {
            "class_name": self.class_name,
            "count": self.count,
            "percentage": self.percentage
        }

class SplitSummaryResponse(BaseModel):
    """Сводка по сплитам версии"""
    id: str = Field(description="ID версии")
    name: str = Field(description="Название версии")
    num_samples: int = Field(description="Количество изображений")
    size_bytes: float = Field(description="Вес версии в байтах")
    overall_balance: float = Field(description="Общий коэффициент баланса классов")
    splits_summary: Dict[str, Dict] = Field(description="Краткая информация по каждому сплиту")
    image_size_stats: Dict[str, Dict] = Field(description="Статистика размеров изображений по сплитам")

class Split(BaseModel):
    class_distribution: List[ClassDistribution] = Field(default_factory=list, description="Информации про каждому классу")

    @property
    def total_samples(self) -> int:
        """Общее количество в сплите"""
        return sum(cd.count for cd in self.class_distribution)
    
    @property
    def num_classes(self) -> int:
        """Количество классов в сплите"""
        return len(self.class_distribution)

    def get_balance_ratio(self) -> float:
        """Коэффициент баланса классов (0-1, где 1 - идеальный баланс)"""
        if not self.class_distribution:
            return 0.0
        
        counts = [cd.count for cd in self.class_distribution]
        min_count = min(counts)
        max_count = max(counts)
        
        return min_count / max_count if max_count > 0 else 0.0
    
    def is_balanced(self, threshold: float = 0.7) -> bool:
        """Проверяет, сбалансирован ли сплит"""
        return self.get_balance_ratio() >= threshold
    
    def get_class_counts(self) -> Dict[str, int]:
        """Словарь {class_name: count}"""
        return {cd.class_name: cd.count for cd in self.class_distribution}
    
    def to_summary(self) -> Dict:
        """Краткая сводка по сплиту (без детальных размеров)"""
        return {
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "balance_ratio": self.get_balance_ratio(),
            "is_balanced": self.is_balanced(),
            "class_distribution": [cd.get_balance_info() for cd in self.class_distribution]
        }
    
    def to_image_size_summary(self) -> Dict:
        """Cтатистика размеров изображений"""
        all_sizes = []
        size_counter = Counter()
        
        for cd in self.class_distribution:
            for size, count in cd.image_size_count.items():
                size_counter[size] += count
                all_sizes.extend([size] * count)
        
        if not size_counter:
            return {"unique_sizes": 0, "total_images": 0}
        
        most_common = size_counter.most_common(1)[0]
        total = sum(size_counter.values())
        
        return {
            "unique_sizes": len(size_counter),
            "total_images": total,
            "most_common_size": most_common[0],
            "most_common_count": most_common[1],
            "size_consistency": round(most_common[1] / total, 2),
            "top_10_sizes": dict(size_counter.most_common(10))
        }