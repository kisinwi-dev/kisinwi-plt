from typing import List
from fastapi import APIRouter, Query, HTTPException

from app.core.crews.praxis_searcher import run_praxis_searcher, PraxisSearchOutput, AGENT_ROLE as PRAXIS_SEARCHER_ROLE
from app.core.crews.ml_models_searcher import run_ml_models_searcher, MLModelsSearcherOutput, AGENT_ROLE as ML_MODELS_SEARCHER_ROLE
from app.core.memory import discussion_context
from app.services.agent_history import agent_history_client

routers = APIRouter(
    prefix='/searcher',
    tags=['searchers']
)

@routers.get(
    "/praxis_in_internet",
    description="Поиск источников практик и их описания.",
    response_model=PraxisSearchOutput
)
def praxis_in_internet(
    discussion_id: str = Query(..., description="ID диалога"),
    search_query: str = Query(..., description="Что именно ищет агент"),
    context: str = Query(..., description="Контекст поиска"),
):
    try:

        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="praxis_searcher", agent_roles=[PRAXIS_SEARCHER_ROLE])

        result = run_praxis_searcher(
            search_query=search_query,
            context=context,
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
    "/info_ml_models",
    description="Получение описания ml моделей разработанных нами.",
    response_model=MLModelsSearcherOutput
)
def get_info_ml_models(
    discussion_id: str = Query(..., description="ID диалога"),
    model_ids: List[str] = Query(..., description="Список ID моделей"),
    context: str = Query(..., description="Контекст"),
):
    try:

        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="ml_models_searcher", agent_roles=[ML_MODELS_SEARCHER_ROLE])

        result = run_ml_models_searcher(
            model_ids=model_ids,
            context=context,
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