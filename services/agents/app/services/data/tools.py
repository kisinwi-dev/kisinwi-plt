

def get_sample_sizes_fot_all_data(dataset_info: dict) -> dict:
    """
    Извлекает информацию о размерах выборки из полной информации о датасете.
    
    Args:
        dataset_info: Полный JSON с информацией о датасете
        
    Returns:
        Словарь с информацией о размерах выборки
    """
    
    result = {
        "version_id": dataset_info["version_id"],
        "total_samples": dataset_info["num_samples"],
        "splits": {}
    }
    
    # Извлекаем информацию по каждому сплиту
    for split_name, split_data in dataset_info["splits"].items():
        class_distribution = split_data["class_distribution"]
        
        # Общее количество в сплите
        total_in_split = sum(cls.get("count", 0) for cls in class_distribution)
        
        # Распределение по классам
        class_counts = {
            cls["class_name"]: cls["count"]
            for cls in class_distribution
        }
        
        result["splits"][split_name] = {
            "total": total_in_split,
            "classes": class_counts
        }
    
    return result