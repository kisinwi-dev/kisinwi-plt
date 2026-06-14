"""Интеграционные тесты предзагрузки весов pretrained-моделей.

Проверяют модуль app.core.models.downloader: разрешение HF-репозитория по имени
timm-модели, скачивание весов в кэш с обновлением статуса задачи и идемпотентность
повторного вызова (взятие из кэша).

ВНИМАНИЕ: тесты ходят в сеть (Hugging Face Hub) и пишут в кэш HF_HOME — это
интеграционные тесты, не unit. Запускать внутри контейнера trainer, где есть timm и
смонтирован кэш ./db/hf_cache.

Запуск напрямую (рекомендуется для trainer, pytest в зависимостях нет):

    docker compose -f docker-compose.yml -f docker-compose.dev.yml \
        run --rm --no-deps -e PYTHONUNBUFFERED=1 trainer python -m tests.test_predownload

Каждая функция test_* синхронная (внутри гоняет async через asyncio.run), поэтому
совместима и с pytest, если он появится в проекте, без pytest-asyncio.
"""
import asyncio
import os
import sys

from app.core.models import predownload_weights
from app.core.models.downloader import resolve_hf_repo

# Публичная timm-модель на HF Hub. Переопределяется через env HF_TEST_MODEL
# (для быстрой проверки удобно взять лёгкую, напр. mobilenetv3_small_050.lamb_in1k).
HF_MODEL = os.getenv("HF_TEST_MODEL", "resnet50")
# Заведомо несуществующее имя — проверка fallback-ветки.
UNKNOWN_MODEL = "no_such_model_xyz_123"


class _RecordingTasker:
    """Заглушка tasker_service: записывает обновления статуса вместо HTTP-запросов."""

    def __init__(self):
        self.updates = []

    async def update_status_task(self, status="running", status_info=None,
                                 percentages=None, error=None, task_id=None):
        self.updates.append({"status_info": status_info, "percentages": percentages})
        return True


def test_resolve_hf_repo_known():
    """HF-hosted модель → возвращается непустой hf_hub_id."""
    hf_hub_id, _ = resolve_hf_repo(HF_MODEL)
    assert hf_hub_id, f"ожидался hf_hub_id для {HF_MODEL}, получено {hf_hub_id!r}"
    assert hf_hub_id.startswith("timm/"), f"неожиданный репозиторий: {hf_hub_id!r}"


def test_resolve_hf_repo_unknown():
    """Несуществующая модель → (None, None), pre-download будет пропущен."""
    assert resolve_hf_repo(UNKNOWN_MODEL) == (None, None)


def test_predownload_downloads_and_caches():
    """predownload_weights скачивает веса, шлёт статусы и кладёт файл в кэш."""
    async def _run():
        tasker = _RecordingTasker()
        result = await predownload_weights(HF_MODEL, tasker, progress_start=6, progress_end=8)

        assert result is True, "ожидался True (модель на HF Hub)"
        # начальный и финальный статусы выставлены
        pcts = [u["percentages"] for u in tasker.updates]
        assert 6 in pcts and 8 in pcts, f"ожидались статусы 6 и 8, получено {pcts}"
        # файл реально в кэше (имя каталога: models--<org>--<repo>)
        hf_hub_id, _ = resolve_hf_repo(HF_MODEL)
        expected = "models--" + hf_hub_id.replace("/", "--")
        hub = os.path.join(os.getenv("HF_HOME", ""), "hub")
        assert expected in os.listdir(hub), \
            f"в кэше {hub} нет {expected}: {os.listdir(hub)}"

    asyncio.run(_run())


def test_predownload_idempotent():
    """Повторный вызов на заполненном кэше снова возвращает True (берёт из кэша)."""
    async def _run():
        tasker = _RecordingTasker()
        result = await predownload_weights(HF_MODEL, tasker, progress_start=6, progress_end=8)
        assert result is True

    asyncio.run(_run())


# --- runner для прямого запуска без pytest -----------------------------------

def _main() -> int:
    print(f"HF_HOME = {os.getenv('HF_HOME')}")
    print(f"HF_HUB_DOWNLOAD_TIMEOUT = {os.getenv('HF_HUB_DOWNLOAD_TIMEOUT')}\n")

    tests = [
        test_resolve_hf_repo_known,
        test_resolve_hf_repo_unknown,
        test_predownload_downloads_and_caches,
        test_predownload_idempotent,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e!r}")

    print(f"\nИтого: {len(tests) - failed}/{len(tests)} прошло")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
