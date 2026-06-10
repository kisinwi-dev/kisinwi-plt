import asyncio
import httpx
from typing import Optional
from pathlib import Path

from app.api.schemas import TaskParams
from app.config import config_services
from app.logs import get_logger

logger = get_logger(__name__)

ML_MODELS_URL = config_services.ML_MODELS['url']

# Общий async-клиент сервиса (живёт всё время работы воркера)
_client = httpx.AsyncClient(timeout=30.0)

async def get_params(
    model_id: str
) -> TaskParams:
    """Получение параметров обучения"""
    try:
        res = await _client.get(f"{ML_MODELS_URL}/models/{model_id}")
        res.raise_for_status()
        params = res.json()['train_params']
        return TaskParams.model_validate(params)

    except httpx.TimeoutException as e:
        raise RuntimeError(f"Таймаут при запросе параметров модели {model_id}") from e

    except httpx.HTTPError as e:
        raise RuntimeError(f"Ошибка при запросе параметров модели {model_id}: {e}") from e

    except (KeyError, ValueError) as e:
        raise RuntimeError(f"Некорректный ответ ml_models для модели {model_id}: {e}") from e

async def patch_model_status(
    model_id: str,
    status: str
):
    """Обновление статуса модели"""
    try:
        res = await _client.patch(
            f"{ML_MODELS_URL}/models/{model_id}",
            json={"status": status}
        )
        res.raise_for_status()

    except httpx.TimeoutException as e:
        raise RuntimeError(f"Таймаут при обновлении статуса модели {model_id}") from e

    except httpx.HTTPError as e:
        raise RuntimeError(f"Ошибка при обновлении статуса модели {model_id}: {e}") from e

async def upload_file_model_in_ml_models(
    model_id: str,
    file_path: str,
    filename: Optional[str] = None
):
    """
    Отправляет файл модели в сервис ml_models.

    Args:
        model_id: ID модели
        file_path: Путь к файлу
        filename: Желаемое имя файла в сервисе (без расширения)
    """
    try:
        path = Path(file_path)

        if filename:
            filename = f"{filename}{path.suffix}"
        else:
            filename = path.name

        url = f"{ML_MODELS_URL}/models/{model_id}/files"

        # Файл может быть большим — читаем в отдельном потоке
        content = await asyncio.to_thread(path.read_bytes)
        files = {'files': (filename, content, 'application/octet-stream')}
        response = await _client.post(url, files=files, timeout=60.0)
        response.raise_for_status()

        file_size = path.stat().st_size / (1024**2)
        logger.info(f"✅ Файл '{filename}' отправлен в ml_models, размер: {file_size:.2f} MB")

    except (httpx.HTTPError, OSError) as e:
        raise RuntimeError(f"Не удалось отправить файл модели {model_id} в ml_models: {e}") from e
