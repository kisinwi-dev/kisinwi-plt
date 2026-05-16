import torch
import onnx
import tempfile
from pathlib import Path
from typing import Tuple

from app.logs import get_logger

logger = get_logger(__name__)

def model_to_onnx(
    model: torch.nn.Module,
    model_id: str,
    input_shape: Tuple[int, ...],
    device: torch.device,
    dynamic_batch: bool = True
) -> str:
    """
    Сохраняет модель в формате ONNX и возвращает путь к файлу.

    Args:
        model: PyTorch модель
        model_id: ID модели для имени файла
        input_shape: Форма входного тензора (batch, channels, height, width)
        device: Устройство модели
        dynamic_batch: Поддержка динамического размера батча

    Returns:
        str: Путь к сохранённому ONNX файлу
    """
    try:
        model.eval()
        model = model.to(device)

        # Создаём случайный вход
        exp_input = torch.randn(*input_shape).to(device)

        # Настройка динамического размера батчей
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Сохранение onnx
        onnx_path = Path(tempfile.gettempdir()) / f"model_{model_id}.onnx"        
        torch.onnx.export(
            model,
            (exp_input,),
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

        logger.info(f"✅ Модель сохранена в ONNX: {onnx_path}")
        
        # Валидация ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX модель валидна")
        
        return str(onnx_path)

    except Exception as e:
        msg = f"Ошибка при сохранении модели в ONNX: {str(e)}"
        logger.error(msg)
        raise Exception(msg) from e
