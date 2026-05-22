from typing import List, Dict
from crewai.tools import BaseTool
from app.services.agent_history.post import agent_history_client
from app.logs import get_logger

logger = get_logger(__name__)


def track_tool(
    agent_role: str, 
    tool: BaseTool, 
    tool_name: str
):
    """
    Обработчик: оборачивает инструмент для отслеживания вызовов.
    
    Args:
        agent_role: Роль агента
        tool: Инструмент для обертки
        tool_name: Название инструмента для логов
    """
    original_run = tool._run
    
    def wrapped_run(*args, **kwargs):
        agent_history_client.tool_call(
            agent_role=agent_role,
            tool_name=tool_name,
            tool_desc=getattr(tool, 'description', None)
        )
        
        try:
            result = original_run(*args, **kwargs)
            agent_history_client.tool_result(
                tool_name=tool_name,
                message=f"Инструмент {tool_name} выполнен"
            )
            return result
        except Exception as e:
            agent_history_client.error(f"Ошибка в {tool_name}: {str(e)}")
            raise
    
    tool._run = wrapped_run
    return tool


def get_tools_with_tracking(
    agent_role: str,
    tools: Dict[str, BaseTool],
) -> List[BaseTool]:
    """
    Возвращает список инструментов с обработчиком.
    
    Args:
        agent_role: Роль агента
        tools: Инструменты, где ключ - название инструмента, а значение инструмент
    
    Returns:
        List[BaseTool]: Список обернутых инструментов
    """    
    
    for name, tool in tools.items():
        tools[name]=track_tool(agent_role, tool, name)
    
    return list(tools.values())