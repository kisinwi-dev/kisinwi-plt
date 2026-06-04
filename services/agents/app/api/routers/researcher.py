from typing import List
from fastapi import APIRouter, Query, HTTPException

from app.core.crews.researcher import run_researcher, ResearcherOutput, AGENT_ROLE as RESEARCHER_ROLE
from app.core.memory import discussion_context
from app.services.agent_history import agent_history_client

routers = APIRouter(
    prefix='/research',
    tags=['researchers']
)

@routers.get(
    "/hypotheses",
    description="Создание гипотез",
    response_model=ResearcherOutput
)
def praxis_in_internet(
    discussion_id: str = Query(..., description="ID диалога"),
    business_requirements: str = Query(..., description="Требования бизнеса"),
    dataset_info: str = Query(..., description="Информация о датасете"),
    denied_hypotheses_info: List[str] = Query(description="Гипотезы отклонённые ML инженером с обоснованием"),
):
    try:

        discussion_context.set(discussion_id)
        agent_history_client.create_discussion(discussion_id, pipeline="researcher", agent_roles=[RESEARCHER_ROLE])

        result = run_researcher(
            business_requirements=business_requirements,
            dataset_info=dataset_info,
            denied_hypotheses_info=denied_hypotheses_info,
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
