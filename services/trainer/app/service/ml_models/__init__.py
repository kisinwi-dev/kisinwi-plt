import requests

from app.api.schemes import TaskParams
from app.config import config_domain
from app.logs import get_logger

logger = get_logger(__name__)

async def get_params(
    model_id: str
) -> TaskParams: 
    """Получение параметров обучения"""
    try:
        res = requests.get(
            f"{config_domain.ML_MODELS}/models/{model_id}",
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
            f"{config_domain.ML_MODELS}/models/{model_id}",
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
