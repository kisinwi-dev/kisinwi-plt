import time
from crewai.types.usage_metrics import UsageMetrics

from app.logs import get_logger
from app.core.crews.analytics import run_data_analysis, run_metrics_analysis
from app.core.crews.ml_engine import run_ml_engineering
from app.core.crews.task_preparer import run_create_task_params_json
from app.services.tasker import tasker

logger = get_logger(__name__)

def full_pipeline(
    dataset_id: str,
    version_id: str,
    bus_req: str,
    count_engine: int = 3,
    verbose: bool = True
) -> dict:
    """
    Полный пайплайн анализа и подготовки задачи
    
    Args:
        dataset_id: ID датасета
        version_id: ID версии датасета
        bus_req: Описание бизнес требований
        count_engine: Количество ML инженеров
        verbose: Подробный вывод в консоли 
    
    Returns:
        Результат выполнения пайплайна с метриками
    """
    
    # Информация по процессам в пайплайне
    pipeline = {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "bus_req": bus_req,
        "count_engine": count_engine,
        "stages": {}
    }
    stages = pipeline["stages"]
    
    try:

        # Анализ датасета 
        info_start_analysis_data = f"{dataset_id} {' версия:' + version_id if version_id else ''}..."
        logger.info(f"Этап 1: Анализ датасета {info_start_analysis_data}")
        
        out_agent_analytic, analysis_metrics = run_data_analysis(
            dataset_id=dataset_id,
            version_id=version_id,
            verbose=verbose
        )
        
        # Фиксируем результат
        stages["data_analysis"] = {
            "out": out_agent_analytic,
            "metrics": _metrics_to_dict(analysis_metrics)
        }
        
        logger.info(f"✅ Анализ завершён")
        
        # ML инженеры
        logger.info(f"Этап 2: Запуск {count_engine} ML инженеров...")
        
        out_engineers, engineers_metrics = run_ml_engineering(
            num_engineers=count_engine,
            info_data=out_agent_analytic,
            verbose=verbose
        )
        
        stages["ml_engineers"] = {
            "out": out_engineers,
            "metrics": _metrics_to_dict(engineers_metrics)
        }
        
        logger.info(f"✅ Получено {len(out_engineers)} предложений от инженеров")
        
        # Суммаризация и превращение в задачу для тренировки
        logger.info(f"Этап 3: Формирование итогового JSON...")

        out_agent_task_preparer, summary_metrics = run_create_task_params_json(
            previous_outputs=out_engineers,
            dataset_id=dataset_id,
            version_id=version_id,
            verbose=verbose
        )
        
        stages["task_preparer"] = {
            "metrics": _metrics_to_dict(summary_metrics)
        }
        logger.info("✅ JSON сформирован")

        logger.info(f"Отправка в сервис задач...")
        task_result = tasker.post_task(out_agent_task_preparer)
        stages["post_task"] = {
            **task_result
        }
        logger.info(f"✅ Задача отправлена")

        logger.info(f"Этап 6: Анализ итогов выполнения задачи")
        
        is_complete = False
        
        logger.info(f"Ожидание выполнения задачи...")

        while is_complete is False:
            is_complete = tasker.task_is_finish(task_result["task_id"])
            time.sleep(1)

        logger.info(f"✅ Задача выполнена. Запускаем анализ полученных метрик")

        out_agent_metrics_analytic, metric_analysis_metrics = run_metrics_analysis(
            task_id=task_result["task_id"],
            dataset_id=dataset_id,
            version_id=version_id,
            bus_req=bus_req,
            verbose=verbose
        )

        stages["metric_analysis"] = {
            "out": out_agent_metrics_analytic,
            "metrics": _metrics_to_dict(metric_analysis_metrics)
        }
        
        logger.info("✨ Пайплайн завершён успешно")

        return {
            **pipeline
        }
        
    except Exception as e:
        pipeline["status"] = "failed"
        pipeline["error"] = str(e)
        
        logger.error(f"❌ Ошибка в пайплайне: {str(e)}")
        
        return {
            "success": False,
            "error": str(e),
            **pipeline
        }


def _metrics_to_dict(
        metrics: UsageMetrics
) -> dict:
    """Конвертирует UsageMetrics в словарь для сериализации"""
    if metrics is None:
        return {"available": False}
    
    return {
        "available": True,
        "total_tokens": metrics.total_tokens,
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "successful_requests": metrics.successful_requests
    }