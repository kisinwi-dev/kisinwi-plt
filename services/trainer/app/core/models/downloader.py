import asyncio
import inspect
import time
from typing import Tuple

import timm
from huggingface_hub import snapshot_download
from tqdm import tqdm as _std_tqdm

from app.logs import get_logger

logger = get_logger(__name__)


class _LoggingTqdm(_std_tqdm):
    """tqdm, который вместо прогресс-бара пишет прогресс в логгер сервиса.

    В non-TTY контейнере обычный прогресс-бар теряется, поэтому пишем % и МБ
    в логи с throttling: не чаще раза в _MIN_INTERVAL секунд или каждые _MIN_STEP %.
    """

    _MIN_INTERVAL = 2.0   # секунд между лог-сообщениями
    _MIN_STEP = 10        # процентов

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_log_t = 0.0
        self._last_log_pct = -100

    def update(self, n=1):
        super().update(n)
        if not self.total:
            return
        pct = int(self.n / self.total * 100)
        now = time.monotonic()
        if (
            now - self._last_log_t >= self._MIN_INTERVAL
            or pct - self._last_log_pct >= self._MIN_STEP
            or pct >= 100
        ):
            mb = self.n / 1024 / 1024
            total_mb = self.total / 1024 / 1024
            logger.info(f"Скачивание весов: {pct}% ({mb:.1f}/{total_mb:.1f} МБ)")
            self._last_log_t = now
            self._last_log_pct = pct


def resolve_hf_repo(model_name: str) -> Tuple[str | None, str | None]:
    """Возвращает (hf_hub_id, hf_hub_filename) для timm-модели.

    Если модель не размещена на HF Hub (legacy torch.hub) или конфиг недоступен —
    возвращает (None, None): это сигнал пропустить pre-download.
    """
    try:
        cfg = timm.get_pretrained_cfg(model_name)
    except Exception as e:
        logger.warning(f"Не удалось получить pretrained_cfg для {model_name}: {e!r}")
        return None, None

    hf_hub_id = getattr(cfg, "hf_hub_id", None)
    hf_hub_filename = getattr(cfg, "hf_hub_filename", None)
    if not hf_hub_id:
        return None, None
    return hf_hub_id, hf_hub_filename


def _do_download(hf_hub_id: str, hf_hub_filename: str | None) -> None:
    """Блокирующая загрузка весов в кэш HF. Прогресс пишется через _LoggingTqdm."""
    kwargs = {"repo_id": hf_hub_id}
    if hf_hub_filename:
        # тянем только нужный файл весов и json-конфиги, без README/картинок
        kwargs["allow_patterns"] = [hf_hub_filename, "*.json"]
    if "tqdm_class" in inspect.signature(snapshot_download).parameters:
        kwargs["tqdm_class"] = _LoggingTqdm
    snapshot_download(**kwargs)


async def predownload_weights(
    model_name: str,
    tasker_service,
    progress_start: int,
    progress_end: int,
) -> bool:
    """Предзагрузка весов pretrained-модели в кэш HF ДО timm.create_model.

    Отдельный этап с видимым прогрессом: после него timm.create_model(pretrained=True)
    возьмёт веса из кэша мгновенно.

    Returns:
        True  — веса скачаны/уже в кэше;
        False — модель не на HF Hub, pre-download пропущен (fallback на torch.hub в get_model).

    При сетевой ошибке/таймауте исключение пробрасывается наверх — статус `failed`
    поставит воркер (report_task_failed).
    """
    hf_hub_id, hf_hub_filename = resolve_hf_repo(model_name)
    if hf_hub_id is None:
        logger.info(
            f"Модель {model_name} не размещена на HF Hub (legacy/torch.hub) — pre-download пропущен."
        )
        return False

    await tasker_service.update_status_task(
        percentages=progress_start,
        status_info="Скачивание весов модели...",
    )
    logger.info(f"Предзагрузка весов {model_name} из HF Hub: {hf_hub_id}")

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, _do_download, hf_hub_id, hf_hub_filename)
    except Exception as e:
        logger.error(f"Ошибка предзагрузки весов {model_name}: {e!r}")
        raise

    await tasker_service.update_status_task(
        percentages=progress_end,
        status_info="Веса модели скачаны.",
    )
    logger.info(f"✅ Веса {model_name} в кэше")
    return True
