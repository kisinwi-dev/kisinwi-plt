from fastapi import APIRouter, Query, HTTPException

from app.core.crews.praxis_searcher import run_praxis_searcher, PraxisSearchOutput

routers = APIRouter(
    prefix='/searcher',
    tags=['searchers']
)

@routers.get(
    "/praxis_in_internet",
    description="Поиск источников практик и их описания.",
    response_model=PraxisSearchOutput
)
def analyze_datasets(
    discussion_id: str = Query(..., description="ID диалога"),
    search_query: str = Query(..., description="Что именно ищет агент"),
    context: str = Query(..., description="Контекст поиска"),
):
    try:
        result = run_praxis_searcher(
            discussion_id=discussion_id,
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