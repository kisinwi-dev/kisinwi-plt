from typing import List
from fastapi import APIRouter, Query, HTTPException

from app.core.crews.researcher import run_researcher, ResearcherOutput, AGENT_ROLE as RESEARCHER_ROLE
from app.services.agent_history import track_discussion

routers = APIRouter(
    prefix='/research',
    tags=['researchers']
)

@routers.get(
    "/hypotheses",
    description="Создание гипотез",
    response_model=ResearcherOutput
)
def create_hypotheses(
    discussion_id: str = Query(..., description="ID диалога"),
    business_requirements: str = Query(..., description="Требования бизнеса"),
    dataset_info: str = Query(..., description="Информация о датасете"),
    denied_hypotheses_info: List[str] = Query(default_factory=list, description="Гипотезы отклонённые ML инженером с обоснованием"),
):
    try:
        with track_discussion(discussion_id, "researcher", "Генерация гипотез", [RESEARCHER_ROLE]):
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
