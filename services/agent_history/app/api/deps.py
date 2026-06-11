from app.core.storage import SystemStorage, ResponseStorage, ToolStorage, DiscussionStorage
from app.core.stream import DiscussionStreamBroker

system_storage = SystemStorage()
response_storage = ResponseStorage()
tool_storage = ToolStorage()
discussion_storage = DiscussionStorage()
discussion_stream_broker = DiscussionStreamBroker()
