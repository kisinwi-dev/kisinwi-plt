from .analytic.crews import run_analysis
from .ml_engin.crews import run_engine_training_pipeline
from .summarizer.crews import run_create_task_params_json
from app.services.tasker import tasker

def full_pipeline_agent(
    dataset_id: str,
    version_id: str,
    task: str,
    count_engine: int,
):
    """
    Полный пайплайн 

    Args:
        dataset_id: id датасета
        version_id: id версии датасета
        task: Класс и тип задачи "NLP/CV/Classic..." + "Сегментация/Классификация..."
        count_engine: Количество используемых агентов-инженеров
    """
    
    analysis_result = run_analysis(dataset_id, version_id).raw

    plans = []

    for index_engine in range(count_engine):
        plans.append(
            run_engine_training_pipeline(task, index_engine, analysis_result).raw
        )

    task_str = run_create_task_params_json(plans).raw
    
    return tasker.post_task(task_str)
