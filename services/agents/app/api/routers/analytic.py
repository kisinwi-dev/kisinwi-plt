from typing import List
from fastapi import APIRouter, Query, HTTPException

from app.core.crews.metrics_analyst import run_metrics_analyst, AGENT_ROLE as METRICS_ANALYST_ROLE
from app.core.crews.dataset_analyst import run_dataset_analyst, AGENT_ROLE as DATASET_ANALYST_ROLE
from app.core.crews.reporter import run_reporter, AGENT_ROLE as REPORTER_ROLE
from app.core.memory import discussion_context, models_context
from app.services.agent_history import agent_history_client

routers = APIRouter(
    prefix='/analytics',
    tags=['analytics']
)

@routers.get(
        "/datasets",
        description="Анализ датасета"
)
def analyze_datasets(
    discussion_id: str = Query(..., description="ID диалога"),
    dataset_id: str = Query(..., description="ID датасета"),
    dataset_version_id: str = Query(..., description="ID версии датасета"),
):
    try:

        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="dataset_analyst", agent_roles=[DATASET_ANALYST_ROLE])

        result = run_dataset_analyst(
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
    finally:
        discussion_context.clear()


@routers.get(
        "/ml_metrics",
        description="Анализ метрик"
)
def analyze_ml_metric(
    discussion_id: str = Query(..., description="ID диалога"),
    business_goal: str = Query(..., description="Требования бизнеса"),
    model_id: str = Query(..., description="ID модели")
):
    try:
        
        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="metrics_analyst", agent_roles=[METRICS_ANALYST_ROLE])

        result = run_metrics_analyst(
            business_goal=business_goal,
            model_id=model_id,
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
    finally:
        discussion_context.clear()

@routers.get(
        "/reporter",
        description="Составление итогов итераций"
)
def reporter(
    discussion_id: str = Query(..., description="ID диалога"),
    models_id: List[str] = Query(..., description="ID моделей используемых для обучения в порядке по итерациям"),
    business_requirements: str = Query(..., description="Требования бизнеса"),
    deployment_constraints: str = Query(..., description="Возможности прода")
):
    try:
        
        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="reporter", agent_roles=[REPORTER_ROLE])
        for model_id in models_id:
            models_context.add_model(model_id)

        result = run_reporter(
            business_requirements=business_requirements,
            deployment_constraints=deployment_constraints,
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )
    finally:
        discussion_context.clear()
