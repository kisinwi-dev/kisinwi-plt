from crewai.types.usage_metrics import UsageMetrics

from app.logs import get_logger
from app.core.crews.analytics import run_analysis
from app.core.crews.ml_engine import run_ml_engineering
from app.core.crews.task_preparer import run_create_task_params_json
from app.services.tasker import tasker

logger = get_logger(__name__)

def full_pipeline(
    dataset_id: str,
    version_id: str | None = None,
    count_engine: int = 3,
    verbose: bool = True
) -> dict:
    """
    Полный пайплайн анализа и подготовки задачи
    
    Args:
        dataset_id: ID датасета
        version_id: ID версии датасета
        count_engine: Количество ML инженеров
        verbose: Подробный вывод в консоли 
    
    Returns:
        Результат выполнения пайплайна с метриками
    """
    
    # Собираем метрики по всем этапам
    pipeline_metrics = {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "stages": {}
    }
    
    try:

        # Анализ датасета 
        
        info_data = f"{dataset_id} {' версия:'+version_id if version_id else ''}..."
        logger.info(f"Этап 1: Анализ датасета {info_data}")
        
        analysis_result, analysis_metrics = run_analysis(
            dataset_id=dataset_id,
            version_id=version_id,
            verbose=verbose
        )
        
        pipeline_metrics["stages"]["analysis"] = {
            "status": "success",
            "metrics": _metrics_to_dict(analysis_metrics)
        }
        
        logger.info(f"✅ Анализ завершён")
        
        # ML инженеры
        
        logger.info(f"Этап 2: Запуск {count_engine} ML инженеров...")
        
        engineers_results, engineers_metrics = run_ml_engineering(
            num_engineers=count_engine,
            analysis_result=analysis_result,
            verbose=verbose
        )
        
        pipeline_metrics["stages"]["ml_engineers"] = {
            "status": "success",
            "count": count_engine,
            "metrics": _metrics_to_dict(engineers_metrics)
        }
        
        logger.info(f"✅ Получено {len(engineers_results)} предложений от инженеров")
        
        # Суммаризация и превращение в задачу для тренировки

        logger.info(f"Этап 3: Формирование итогового JSON...")
        
        final_json, summary_metrics = run_create_task_params_json(
            previous_outputs=engineers_results,
            verbose=verbose
        )
        
        pipeline_metrics["stages"]["summary"] = {
            "status": "success",
            "metrics": _metrics_to_dict(summary_metrics)
        }
        
        logger.info("✅ JSON сформирован")
        
        # Отправка в сервис задач
        logger.info(f"Этап 4: Отправка в сервис задач...")
        
        task_result = tasker.post_task(final_json)
        
        pipeline_metrics["stages"]["tasker"] = {
            "status": "success",
            "task_id": task_result.get("task_id") if isinstance(task_result, dict) else None
        }
        
        logger.info(f"✅ Задача отправлена")
        
        # Общая сводка
        total_tokens = sum([
            analysis_metrics.total_tokens if analysis_metrics else 0,
            engineers_metrics.total_tokens if engineers_metrics else 0,
            summary_metrics.total_tokens if summary_metrics else 0
        ])
        
        pipeline_metrics["summary"] = {
            "status": "success",
            "total_tokens": total_tokens,
            "total_requests": (
                (analysis_metrics.successful_requests if analysis_metrics else 0) +
                (engineers_metrics.successful_requests if engineers_metrics else 0) +
                (summary_metrics.successful_requests if summary_metrics else 0)
            )
        }
        
        logger.info('✨ Пайплайн завершён успешно')
        
        return {
            "success": True,
            "result": final_json,
            "task_response": task_result,
            "metrics": pipeline_metrics
        }
        
    except Exception as e:
        pipeline_metrics["status"] = "failed"
        pipeline_metrics["error"] = str(e)
        
        logger.error(f"\n❌ Ошибка в пайплайне: {str(e)}")
        
        return {
            "success": False,
            "error": str(e),
            "metrics": pipeline_metrics
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