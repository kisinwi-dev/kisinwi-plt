from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.schemas import (
    SystemMessageType, HealthResponse, 
    ToolStatus, AgentStatus
)
from app.core.health import check_health_all

routers = APIRouter(
    prefix="/info", 
    tags=["info"]
)

@routers.get(
    "/health",
    summary="Проверка работоспособности сервиса",
    description="Проверяет работу сервиса и зависимостей",
    response_model=HealthResponse,
    responses={
        200: {"description": "Выдаёт информацию о состояниях"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def health():
    """Проверка подключения к БД"""
    try:
        return check_health_all()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )

@routers.get(
    "/type_system_mes",
    summary="Типы информативных сообщений",
    description="Получить список возможных типов информативных сообщений от сервиса агентов",
    responses={
        200: {"description": "Получен список типов"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def get_type_mes_system():
    """Получаем список возможных типов сообщений от системы агентов"""
    try:
        types = [t.value for t in SystemMessageType]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"types": types}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )
    
@routers.get(
    "/status_tool",
    summary="Статусы инструментов",
    description="Получить список возможных статусов инструментов",
    responses={
        200: {"description": "Получен список статусов"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def get_type_tool():
    """Получаем статусы инструментов"""
    try:
        types = [t.value for t in ToolStatus]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"types": types}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )

@routers.get(
    "/status_agent",
    summary="Статусы агентов",
    description="Получить список возможных статусов агентов",
    responses={
        200: {"description": "Получен список статусов"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def get_type_agent():
    """Получаем статусы агентов"""
    try:
        types = [t.value for t in AgentStatus]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"types": types}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )