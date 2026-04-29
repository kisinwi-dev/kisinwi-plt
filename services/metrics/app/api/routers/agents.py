from fastapi import APIRouter, HTTPException, Depends
from app.api.schemes import AgentResponse, ConversationResponses
from app.api.deps import get_agent_metrics_manager, AgentsResponseManager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/response/add")
async def add_agent_response(
    response: AgentResponse,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Добавление нового ответа агента
    """
    try:
        manager.add_response(response)
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}")
async def get_conversation_responses(
    conversation_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Получение всех ответов для конкретного диалога
    """
    try:
        responses = manager.get_conversation_responses(conversation_id)
        
        return ConversationResponses(
            conversation_id=conversation_id,
            responses=responses
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/response/{response_id}", response_model=AgentResponse)
async def get_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Получение ответа агента по ID
    """
    try:
        response = manager.get_response_by_id(response_id)
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Удаление всех ответов диалога
    """
    try:
        result = manager.delete_conversation(conversation_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/response/{response_id}")
async def delete_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Удаление ответа агента
    """
    try:
        manager.delete_response(response_id)
        
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/exists", response_model=dict)
async def check_conversation_exists(
    conversation_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Проверка существования диалога
    """
    try:
        exists = manager.conversation_exists(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "exists": exists
        }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
