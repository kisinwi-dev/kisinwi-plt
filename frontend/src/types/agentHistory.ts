// Типы для сервиса истории агентов (agent_history).
// Зеркалят Pydantic-схемы бэкенда: services/agent_history/app/api/schemas/.

// Статус дискуссии.
export type DiscussionStatus = 'active' | 'completed' | 'failed';

// Статус запуска агента / инструмента (значения как у бэкенда — строки с пробелом).
export type AgentStatus = 'IN PROGRESS' | 'SUCCEED' | 'ERROR';
export type ToolStatus = 'IN PROGRESS' | 'SUCCEED' | 'ERROR';

// Тип системного сообщения.
export type SystemMessageType = 'INFO' | 'WARNING' | 'ERROR';

// Агент дискуссии и использованные им модели LLM.
export interface AgentModelInfo {
  role: string;
  models: string[];
}

// Метаданные дискуссии.
export interface DiscussionMeta {
  discussion_id: string;
  title: string | null;
  status: DiscussionStatus;
  tags: string[];
  pipeline: string | null;
  agent_roles: string[];
  created_at: string;
  finished_at: string | null;
  // Вычисляемые агрегаты (присутствуют в ответе списка дискуссий).
  responses_count?: number;
  tool_calls_count?: number;
  agents?: AgentModelInfo[];
}

// Ответ (запуск) агента.
export interface AgentResponse {
  response_id: string;
  status: AgentStatus;
  agent_role: string;
  text: string;
  timestamp: string;
  model: string | null;
  duration_ms: number | null;
  task_name: string | null;
  iteration: number | null;
}

// Системное сообщение.
export interface SystemMessage {
  type_: SystemMessageType;
  message: string;
  timestamp: string;
}

// Вызов инструмента в рамках запуска агента.
export interface Tool {
  id: string;
  agent_role: string;
  name: string;
  status: ToolStatus;
  message: string;
  timestamp: string;
  input_args: Record<string, unknown> | null;
  output: unknown;
  duration_ms: number | null;
  error_traceback: string | null;
  response_id: string | null;
}

// Параметры фильтрации списка дискуссий.
export interface DiscussionListParams {
  status?: DiscussionStatus;
  pipeline?: string;
  skip?: number;
  limit?: number;
}
