from crewai.types.usage_metrics import UsageMetrics

from .analytic import new_analytic_reporter, new_task_analytic
from .ml_engin import new_agent_ml_engineer, new_task_search_best_model
from .summarizer import new_agent_task_preparer, new_task_summary
from app.services.tasker import tasker

def full_pipeline_agent(
    dataset_id: str,
    version_id: str,
    count_engine: int,
):
    """
    Полный пайплайн 

    Args:
        dataset_id: id датасета
        version_id: id версии датасета
        count_engine: Количество используемых агентов-инженеров
    """

    agent_analyst = new_analytic_reporter()
    agent_enginers = [
        new_agent_ml_engineer(i) for i in range(count_engine)
    ]
    agent_task_prepare = new_agent_task_preparer()
   
    new_task_analytic(dataset_id, version_id)
   

    return tasker.post_task(result.raw)
