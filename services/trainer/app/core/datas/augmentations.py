import torchvision.transforms as transforms
from typing import Dict, Any, List
from app.logs import get_logger

logger = get_logger(__name__)

# Список допустимых типов аугментации
ALLOWED_TRANSFORMS = {
    # Геометрические трансформации
    "Resize": transforms.Resize,
    "CenterCrop": transforms.CenterCrop,
    "RandomCrop": transforms.RandomCrop,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "FiveCrop": transforms.FiveCrop,
    "TenCrop": transforms.TenCrop,
    "Pad": transforms.Pad,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomVerticalFlip": transforms.RandomVerticalFlip,
    "RandomRotation": transforms.RandomRotation,
    "RandomAffine": transforms.RandomAffine,
    "RandomPerspective": transforms.RandomPerspective,

    # Цветовые трансформации
    "ColorJitter": transforms.ColorJitter,
    "Grayscale": transforms.Grayscale,
    "RandomGrayscale": transforms.RandomGrayscale,
    "RandomAutocontrast": transforms.RandomAutocontrast,
    "RandomEqualize": transforms.RandomEqualize,
    "RandomPosterize": transforms.RandomPosterize,
    "RandomAdjustSharpness": transforms.RandomAdjustSharpness,
    "RandomInvert": transforms.RandomInvert,
    "RandomSolarize": transforms.RandomSolarize,

    # Размытие и шум
    "GaussianBlur": transforms.GaussianBlur,
    "RandomApply": transforms.RandomApply,

    # Преобразования тензоров
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
    "ToPILImage": transforms.ToPILImage,
    "ConvertImageDtype": transforms.ConvertImageDtype,

    # Продвинутые
    "AutoAugment": transforms.AutoAugment,
    "RandAugment": transforms.RandAugment,
    "TrivialAugmentWide": transforms.TrivialAugmentWide,
    "AugMix": transforms.AugMix,
}

    
def build_transforms(transforms_config: List[Dict[str, Any]]) -> transforms.Compose:
    """
    Создаёт transforms.Compose из конфигурации

    Args:
        transforms_config: Список конфигураций трансформаций
            Пример: [
                {"name": "RandomResizedCrop", "params": {"size": [224, 224], "scale": [0.7, 1.0]}},
                {"name": "RandomHorizontalFlip", "params": {"p": 0.5}},
                {"name": "ToTensor", "params": {}},
                {"name": "Normalize", "params": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
            ]

    Returns:
        transforms.Compose объект
    """
    transform_list = []

    for config in transforms_config:
        transform_name = config.get("name")

        if not transform_name:
            logger.warning(f"Пропущена неизвестная трансформация: {config}")
            continue

        if transform_name not in ALLOWED_TRANSFORMS:
            logger.warning(
                f"Трансформация '{transform_name}' не найдена среди доступных. "
                f"Доступные: {list(ALLOWED_TRANSFORMS.keys())}"
            )
            continue

        params = config.get("params", {})


        try:
            transform_class = ALLOWED_TRANSFORMS[transform_name]

            # Обработка параметров
            params = _process_special_params(transform_name, params)

            # создание экземпляра трансформации
            transform_list.append(transform_class(**params))

            logger.debug(f"✅ Добавлена трансформация: {transform_name} с параметрами {params}")

        except Exception as e:
            logger.error(f"❌ Ошибка создания трансформации '{transform_name}': {e}")
            raise ValueError(f"Не удалось создать трансформацию '{transform_name}': {e}")

    if not transform_list:
        raise ValueError("Не создано ни одной трансформации")

    return transforms.Compose(transform_list)


def _process_special_params(
    transform_name: str, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Обрабатывает специальные параметры для конкретных трансформаций
    """
    processed_params = params.copy()

    # преобразование size из списка в tuple если нужно
    if 'size' in processed_params and isinstance(processed_params['size'], list):
        processed_params['size'] = tuple(processed_params['size'])

    # преобразование scale из списка в tuple
    if 'scale' in processed_params and isinstance(processed_params['scale'], list):
        processed_params['scale'] = tuple(processed_params['scale'])
    
    # преобразование ratio из списка в tuple
    if 'ratio' in processed_params and isinstance(processed_params['ratio'], list):
        processed_params['ratio'] = tuple(processed_params['ratio'])

    return processed_params
