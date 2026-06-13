// Типы для сервиса истории агентов (agent_history).
// Зеркалят Pydantic-схемы бэкенда: services/agent_history/app/api/schemas/.

// Статус дискуссии.
export type DiscussionStatus = 'active' | 'completed' | 'failed';

// Человекочитаемые подписи статусов дискуссии.
export const DISCUSSION_STATUS_LABELS: Record<DiscussionStatus, string> = {
  active: 'Активна',
  completed: 'Завершена',
  failed: 'Ошибка',
};

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

// Человекочитаемые названия пайплайнов; для неизвестных показываем сырое значение.
const PIPELINE_LABELS: Record<string, string> = {
  development: 'Полный цикл',
  quick_training: 'Быстрый прогон',
};

export const getPipelineLabel = (pipeline: string): string =>
  PIPELINE_LABELS[pipeline] ?? pipeline;

// Заголовок дискуссии: название → пайплайн → запасной вариант.
export const getDiscussionTitle = (d: Pick<DiscussionMeta, 'title' | 'pipeline'>): string =>
  d.title ?? (d.pipeline ? getPipelineLabel(d.pipeline) : null) ?? 'Без названия';

// Список агентов: из агрегата, иначе из заявленных ролей meta.
export const getDiscussionAgents = (
  d: Pick<DiscussionMeta, 'agents' | 'agent_roles'>,
): AgentModelInfo[] => d.agents ?? d.agent_roles.map(role => ({ role, models: [] }));

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
