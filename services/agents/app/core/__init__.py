import uuid

from app.logs import get_logger
from app.core.crews.analytics import run_data_analysis, run_metrics_analysis
from app.core.crews.ml_engine import run_ml_engineering
from app.core.crews.task_preparer import run_create_task_params_json
from app.core.crews.work_with_ml_models import run_create_desc_model
from app.services.tasker import tasker
from app.services.ml_models import ml_models
from app.services.data import get_dataset_info_classes

logger = get_logger(__name__)

def full_pipeline(
    dataset_id: str,
    dataset_version_id: str,
    model_name: str,
    bus_req: str,
    count_engine: int = 3,
    iterations: int = 2,
    verbose: bool = False
) -> dict:
    """
    Полный пайплайн анализа и подготовки задачи
    
    Args:
        dataset_id: ID датасета
        dataset_version_id: ID версии датасета
        bus_req: Описание бизнес требований
        count_engine: Количество ML инженеров
        verbose: Подробный вывод в консоли 
    
    Returns:
        Результат выполнения пайплайна с метриками
    """
    
    discussion_id = str(uuid.uuid4())

    try:
        logger.info(f"Запуск диалога: {discussion_id}")
        # Анализ датасета 
        info_start_analysis_data = f"{dataset_id}{' версия: ' + dataset_version_id if dataset_version_id else ''} ..."
        logger.info(f"Этап 1: Анализ датасета {info_start_analysis_data}")
        
        out_agent_analytic, _ = run_data_analysis(
            dataset_id=dataset_id,
            version_id=dataset_version_id,
            conversation_id=discussion_id,
            verbose=verbose
        )
        out_agent_analytic = "Хорошие даннные. Классификация кошек и собак. Сбалансированные данные. 1080х720 изображения"
        logger.info(f"✅ Анализ завершён")

        ml_models_ids = []
        version = 0

        for iter in range(1, iterations+1):
            
            version += 1
            
            # ML инженеры
            logger.info(f"Этап [{iter}|2]: Запуск {count_engine} ML инженеров...")
            out_engineers, _ = run_ml_engineering(
                num_engineers=count_engine,
                conversation_id=discussion_id,
                info_data=out_agent_analytic,
                ml_models_ids=ml_models_ids,
                verbose=verbose
            )
            logger.info(f"✅ Получено {len(out_engineers)} предложений от инженеров")

            # Суммаризация и превращение в задачу для тренировки
            logger.info(f"Этап [{iter}|3]: Формирование этапов обучения...")
            train_params, _ = run_create_task_params_json(
                previous_outputs=out_engineers,
                dataset_id=dataset_id,
                version_id=dataset_version_id,
                conversation_id=discussion_id,
                verbose=verbose
            )
            logger.info("✅ План обучения сформирован")

            logger.info(f"Этап [{iter}|4]: Создание задачи и формирование модели...")

            models_desc, _ = run_create_desc_model(
                train_params,
                out_agent_analytic,
                verbose=verbose
            )

            model_id = ml_models.create_model(
                name=model_name,
                version=version,
                model_type=models_desc.model_type,
                description=models_desc.description,
                classes=get_dataset_info_classes(dataset_id),
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                train_params=train_params
            )
            ml_models_ids.append(model_id)
            
            logger.info(f"✅ Информация о модели занесена")

            task_id = tasker.task_training_create(
                task_name=f"Тренировка модели {model_name}",
                model_id=model_id,
                discussion_id=discussion_id
            )
            logger.info(f"✅ Задача создана")

            logger.info(f"Этап [{iter}|5]: Анализ итогов выполнения задачи")
            logger.info(f"Ожидание выполнения задачи...")
            
            is_completed = False
            while not is_completed:
                is_completed, task = tasker.waiting_completed(task_id)
                if is_completed:
                    break
                else:
                    logger.info(f"❗ Ошибка при выполнении задачи")
                    logger.info(f"Пересматриваем параметры обучения на основе полученной ошибки")
                    train_params, _ = run_create_task_params_json(
                        previous_outputs=out_engineers,
                        dataset_id=dataset_id,
                        version_id=dataset_version_id,
                        conversation_id=discussion_id,
                        extra=f"Ранее была получена ошибка: Error: {task['error_message']}",
                        verbose=verbose
                    )
                    ml_models.update_model(model_id, train_params)
                    logger.info(f"✅ Параметры модели изменены")

                    task_id = tasker.task_training_create(
                        task_name=f"Тренировка модели {model_name}",
                        model_id=model_id,
                        discussion_id=discussion_id
                    )
                    logger.info(f"✅ Задача создана")


            
            logger.info(f"✅ Задача выполнена. Запускаем анализ полученных метрик")

            ml_model_merics_desc, _ = run_metrics_analysis(
                model_id=model_id,
                dataset_id=dataset_id,
                version_id=dataset_version_id,
                bus_req=bus_req,
                conversation_id=discussion_id,
                verbose=verbose
            )
            logger.info(f"Итерация рассуждений №{iter} завершена, в процессе была обучена модель '{model_id}'")
        
        logger.info("✨ Пайплайн завершён успешно")

        return {
            "success": True
        }

    except Exception as e:

        logger.error(f"❌ Ошибка в пайплайне: {str(e)}")
        
        return {
            "success": False,
            "error": str(e)
        }
