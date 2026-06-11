import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ModelMetricAdd,
    ModelMetricAdds,
    ModelMetrics,
    ModelMetricsBatchRequest,
    ModelMetricsSummary,
    ModelsCompareRequest,
    ModelsCompareResponse,
)
from app.api.deps import (
    get_cv_training_metrics_manager,
    CVMetricManager,
    get_metric_stream_broker,
    MetricStreamBroker,
)
from app.core.stats import compute_model_summary, compare_models
from app.core.stream import format_sse
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["metrics"])

# Интервал keepalive-комментариев SSE, чтобы соединение не закрывали прокси
KEEPALIVE_INTERVAL_S = 15.0

@router.post(
    "/add",
    summary="Добавить метрику модели",
    description="Добавляет значения одной метрики модели в её выборку (train/val/test); "
                "выборка берётся из поля split или из префикса названия; "
                "при отсутствии модели создаёт запись",
    response_description="Статус операции",
)
async def add_metric(
    metric: ModelMetricAdd,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager),
    broker: MetricStreamBroker = Depends(get_metric_stream_broker),
):
    success = manager.add_metric(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрики")
    broker.publish(metric.model_id)
    return {"status": "ok"}

@router.post(
    "/adds",
    summary="Добавить несколько метрик модели",
    description="Добавляет значения сразу нескольких метрик модели за один запрос (используется при обучении); "
                "каждая метрика попадает в свою выборку (train/val/test)",
    response_description="Статус операции",
)
async def add_metrics(
    metric: ModelMetricAdds,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager),
    broker: MetricStreamBroker = Depends(get_metric_stream_broker),
):
    success = manager.add_metrics(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрик")
    broker.publish(metric.model_id)
    return {"status": "ok"}

@router.post(
    "/batch",
    response_model=List[ModelMetrics],
    summary="Получить метрики нескольких моделей",
    description="Возвращает метрики сразу нескольких моделей за один запрос, разбитые по выборкам "
                "(train/val/test); модели без метрик в ответ не попадают",
    response_description="Список метрик по каждой найденной модели",
)
async def get_models_metrics(
    body: ModelMetricsBatchRequest,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    return manager.get_models_metrics(body.model_ids)

@router.post(
    "/compare",
    response_model=ModelsCompareResponse,
    summary="Сравнить модели по метрикам",
    description="Сравнивает несколько моделей (минимум 2) по метрикам выбранной выборки "
                "(по умолчанию val): лучшее и финальное значения, эпоха лучшего значения, "
                "лидер и отставание от него. Направление метрики определяется по названию: "
                "loss/error-подобные — чем меньше, тем лучше, остальные — чем больше. "
                "Модели без сохранённых метрик попадают в missing (это не ошибка)",
    response_description="Таблица сравнения моделей по каждой метрике",
)
async def compare_models_metrics(
    body: ModelsCompareRequest,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    all_metrics = manager.get_models_metrics(body.model_ids)
    return compare_models(all_metrics, body.model_ids, body.split, body.metrics)

@router.get(
    "/{model_id}",
    response_model=ModelMetrics,
    summary="Получить метрики модели",
    description="Возвращает все метрики указанной модели с их значениями по эпохам, "
                "разбитые по выборкам (train/val/test)",
    response_description="Метрики модели",
)
async def get_model_metrics(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    metrics = manager.get_model_metrics(model_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Модель {model_id} не найдена")
    return metrics

@router.get(
    "/{model_id}/stream",
    summary="Поток метрик модели (SSE)",
    description="Server-Sent Events: при подключении отправляет событие metrics с текущим "
                "снимком метрик модели (схема ModelMetrics), затем шлёт такое же событие "
                "при каждом добавлении метрик (раз в эпоху от trainer). "
                "Каждые 15 секунд отправляется keepalive-комментарий. "
                "Для модели без метрик отдаётся пустой снимок — подключаться можно до первой эпохи. "
                "Сервис не знает о завершении обучения: поток открыт до отключения клиента",
    response_description="Поток событий text/event-stream",
)
async def stream_model_metrics(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager),
    broker: MetricStreamBroker = Depends(get_metric_stream_broker),
):
    def snapshot() -> str:
        metrics = manager.get_model_metrics(model_id) or ModelMetrics(model_id=model_id)
        return format_sse("metrics", metrics.model_dump_json())

    async def event_generator():
        queue = broker.subscribe(model_id)
        try:
            yield snapshot()
            while True:
                try:
                    await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL_S)
                    # Схлопываем накопившиеся уведомления — снимок всё равно полный
                    while not queue.empty():
                        queue.get_nowait()
                    yield snapshot()
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            broker.unsubscribe(model_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get(
    "/{model_id}/summary",
    response_model=ModelMetricsSummary,
    summary="Сводная статистика метрик модели",
    description="Возвращает по каждой метрике каждой выборки финальное и лучшее значения, "
                "эпоху лучшего значения, min/max и число эпох, а также разрывы train/val "
                "на последней эпохе (gap > 0 — на валидации хуже, признак переобучения). "
                "Направление метрики определяется по названию: loss/error-подобные — "
                "чем меньше, тем лучше, остальные — чем больше",
    response_description="Сводка метрик модели по выборкам",
)
async def get_model_metrics_summary(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    metrics = manager.get_model_metrics(model_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Модель {model_id} не найдена")
    return compute_model_summary(metrics)

@router.get(
    "/{model_id}/exists",
    summary="Проверить наличие метрик модели",
    description="Проверяет, есть ли в системе сохранённые метрики для указанной модели",
    response_description="Флаг наличия метрик модели",
)
async def model_metrics_exists(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    exists = manager.model_metrics_exists(model_id)
    return {"model_id": model_id, "exists": exists}


@router.delete(
    "/{model_id}",
    summary="Удалить метрики модели",
    description="Удаляет все сохранённые метрики указанной модели из системы",
    response_description="Идентификатор модели и признак удаления",
)
async def delete_model_metrics(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager),
    broker: MetricStreamBroker = Depends(get_metric_stream_broker),
):
    deleted = manager.delete_metric(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Метрики модели {model_id} не найдены")
    broker.publish(model_id)
    return {"model_id": model_id, "deleted": True}
