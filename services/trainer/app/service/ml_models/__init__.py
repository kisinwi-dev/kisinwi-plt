import requests
from typing import Optional
from pathlib import Path

from app.api.schemas import TaskParams
from app.config import config_services
from app.logs import get_logger

logger = get_logger(__name__)

ML_MODELS_URL = config_services.ML_MODELS['url']

async def get_params(
    model_id: str
) -> TaskParams: 
    """Получение параметров обучения"""
    try:
        res = requests.get(
            f"{ML_MODELS_URL}/models/{model_id}",
            timeout=30
        )
        res.raise_for_status()
        params = res.json()['train_params']
        return TaskParams.model_validate(params)

    except requests.exceptions.Timeout as e:
        mes = f"Таймаут при запросе параметров модели {model_id}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except requests.exceptions.ConnectionError as e:
        mes = f"Ошибка соединения при запросе модели {model_id}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except requests.exceptions.RequestException as e:
        mes = f"Ошибка при запросе параметров модели {model_id}: {str(e)}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except ValueError as e:
        mes = f"Ошибка парсинга JSON для модели {model_id}: {str(e)}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except Exception as e:
        mes = f"Неожиданная ошибка при получении параметров модели {model_id}: {str(e)}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

async def path_status_model(
    model_id: str,
    status: str
): 
    """Получение параметров обучения"""
    try:

        data = {
            "status": status
        }

        res = requests.patch(
            f"{ML_MODELS_URL}/models/{model_id}",
            json=data,
            timeout=30
        )
        res.raise_for_status()

    except requests.exceptions.Timeout as e:
        mes = f"Таймаут при обновлении статуса модели {model_id}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except requests.exceptions.ConnectionError as e:
        mes = f"Ошибка соединения при обновлении статуса модели {model_id}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except requests.exceptions.RequestException as e:
        mes = f"Ошибка при обновлении статуса модели {model_id}: {str(e)}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

    except Exception as e:
        mes = f"Неожиданная ошибка при обновлении статуса модели {model_id}: {str(e)}"
        logger.error(mes, exc_info=True)
        raise Exception(mes) from e

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

        with open(file_path, 'rb') as f:
            files = {'files': (filename, f, 'application/octet-stream')}
            response = requests.post(url, files=files, timeout=60)

        response.raise_for_status()
        file_size = path.stat().st_size / (1024**2)
        logger.info(f"✅ Файл '{filename}' отправлен в ml_models, размер: {file_size:.2f} MB")

    except Exception as e:
        logger.error(f"❌ Ошибка при отправке файла: {e}")
        return False