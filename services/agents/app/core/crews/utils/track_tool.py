import time
import traceback
from uuid import uuid4
from typing import List
from crewai.tools import BaseTool
from app.services.agent_history import agent_history_client
from app.logs import get_logger

logger = get_logger(__name__)


def track_tool(agent_role: str, tool: BaseTool) -> BaseTool:
    original_run = tool._run

    def wrapped_run(*args, **kwargs):
        id_tool = str(uuid4())
        start_time = time.time()

        agent_history_client.tool_start(
            id=id_tool,
            agent_role=agent_role,
            name=tool.name,
            message=getattr(tool, 'description', None),
            input_args=kwargs if kwargs else None,
        )

        try:
            result = original_run(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            agent_history_client.tool_succed(
                id=id_tool,
                agent_role=agent_role,
                name=tool.name,
                message=f"Инструмент {tool.name} завершил работу",
                output=result,
                duration_ms=duration_ms,
            )
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            agent_history_client.tool_error(
                id=id_tool,
                agent_role=agent_role,
                name=tool.name,
                message=f"Ошибка: {str(e)}",
                error_traceback=traceback.format_exc(),
                duration_ms=duration_ms,
            )
            raise

    tool._run = wrapped_run
    return tool


def get_tools_with_tracking(
    agent_role: str,
    tools: List[BaseTool],
) -> List[BaseTool]:
    return [track_tool(agent_role, tool) for tool in tools]