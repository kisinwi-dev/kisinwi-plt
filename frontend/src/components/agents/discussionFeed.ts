// Единая лента дискуссии и производные индикаторы активности.
// Загрузка ленты живёт здесь (а не в DiscussionView), потому что её данные нужны
// сразу в двух местах: внизу — сама лента (DiscussionView), сверху — индикаторы
// активного агента и состояния обучения (DiscussionInfo). Один источник правды.

import { agentHistoryService } from '../../services/agentHistoryService';
import { taskerService } from '../../services/taskerService';
import type { AgentResponse, SystemMessage } from '../../types/agentHistory';
import type { TrainingTask } from '../../types/tasks';
import { parseBackendDate } from '../../utils/format';

// Элемент единой ленты: ответ агента, системное сообщение или этап обучения.
export type FeedItem =
  | { kind: 'response'; timestamp: string; data: AgentResponse }
  | { kind: 'system'; timestamp: string; data: SystemMessage }
  | { kind: 'training'; timestamp: string; data: TrainingTask };

// Состояние сервиса обучения (по последней задаче дискуссии).
export type TrainingState = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

// Сводка обучения для карточки «Сервис обучения» в секции исполнителей.
// total учитывает несколько последовательных обучений (модель → ещё модель).
export interface TrainingSummary {
  state: TrainingState;
  total: number;
  currentName: string | null;
}

// Грузим ленту дискуссии: ответы агентов + системные сообщения + задачи обучения,
// мёржим в единый список и сортируем по времени (по возрастанию).
export async function loadDiscussionFeed(discussionId: string): Promise<FeedItem[]> {
  const [responses, systemMessages, { tasks }] = await Promise.all([
    agentHistoryService.getResponses(discussionId),
    agentHistoryService.getSystemMessages(discussionId),
    taskerService.getTasks({ discussion_id: discussionId }),
  ]);
  const items: FeedItem[] = [
    ...responses.map((data): FeedItem => ({ kind: 'response', timestamp: data.timestamp, data })),
    ...systemMessages.map((data): FeedItem => ({ kind: 'system', timestamp: data.timestamp, data })),
    ...tasks.map((data): FeedItem => ({ kind: 'training', timestamp: data.created_at, data })),
  ];
  items.sort((a, b) => (parseBackendDate(a.timestamp) ?? 0) - (parseBackendDate(b.timestamp) ?? 0));
  return items;
}

// Роль агента, который сейчас работает: последний по времени запуск со статусом
// IN PROGRESS. Лента отсортирована по возрастанию — идём с конца.
export function deriveActiveAgentRole(feed: FeedItem[]): string | null {
  for (let i = feed.length - 1; i >= 0; i--) {
    const item = feed[i];
    if (item.kind === 'response' && item.data.status === 'IN PROGRESS') {
      return item.data.agent_role;
    }
  }
  return null;
}

// Сводка обучения по задачам дискуссии. Состояние берём у ПОСЛЕДНЕЙ задачи
// (по порядку ленты): waiting/running → 'running', иначе её статус; total —
// сколько обучений всего (несколько моделей подряд); нет задач → idle/0.
export function deriveTrainingSummary(feed: FeedItem[]): TrainingSummary {
  const tasks: TrainingTask[] = [];
  for (const item of feed) {
    if (item.kind === 'training') tasks.push(item.data);
  }
  if (tasks.length === 0) return { state: 'idle', total: 0, currentName: null };

  const current = tasks[tasks.length - 1];
  const status = current.status;
  let state: TrainingState = 'idle';
  if (status === 'waiting' || status === 'running') state = 'running';
  else if (status === 'completed' || status === 'failed' || status === 'cancelled') state = status;

  return { state, total: tasks.length, currentName: current.name ?? null };
}
