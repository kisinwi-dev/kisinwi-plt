import React from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { AgentResponse, SystemMessage, SystemMessageType } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks';
import { POLL_INTERVAL_DISCUSSION_MS } from '../../constants';
import { formatDateTime } from '../../utils/format';
import MessageBubble from './MessageBubble';

interface Props {
  discussionId: string;
  // Если дискуссия активна — лента периодически обновляется (polling прогресса).
  active?: boolean;
}

// Элемент единой ленты: ответ агента или системное сообщение.
type FeedItem =
  | { kind: 'response'; timestamp: string; data: AgentResponse }
  | { kind: 'system'; timestamp: string; data: SystemMessage };

// Иконка системного сообщения по типу.
const SYSTEM_ICONS: Record<SystemMessageType, string> = {
  INFO: 'fa-circle-info',
  WARNING: 'fa-triangle-exclamation',
  ERROR: 'fa-circle-exclamation',
};

const DiscussionView: React.FC<Props> = ({ discussionId, active = false }) => {
  const { showNotification } = useNotification();

  const { data, loading } = usePolling<FeedItem[]>(
    async () => {
      const [responses, systemMessages] = await Promise.all([
        agentHistoryService.getResponses(discussionId),
        agentHistoryService.getSystemMessages(discussionId),
      ]);
      const items: FeedItem[] = [
        ...responses.map((data): FeedItem => ({ kind: 'response', timestamp: data.timestamp, data })),
        ...systemMessages.map((data): FeedItem => ({ kind: 'system', timestamp: data.timestamp, data })),
      ];
      items.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      return items;
    },
    {
      intervalMs: POLL_INTERVAL_DISCUSSION_MS,
      // Ленту грузим всегда хотя бы раз; повторяем опрос только пока дискуссия активна.
      continueWhile: () => active,
      onError: err =>
        showNotification(err instanceof Error ? err.message : 'Ошибка загрузки диалога', 'error'),
      // active в deps: когда статус дискуссии становится active (meta догрузилась),
      // цикл опроса перезапускается, иначе он умер бы после первого запроса.
      deps: [discussionId, active],
    },
  );

  const feed = data ?? [];

  if (loading && feed.length === 0) {
    return <div className="loading-state">Загрузка диалога...</div>;
  }

  if (feed.length === 0 && !active) {
    return <p className="empty-state">В этой дискуссии пока нет сообщений.</p>;
  }

  return (
    <div className="discussion-feed">
      {feed.map(item =>
        item.kind === 'response' ? (
          <MessageBubble key={item.data.response_id} discussionId={discussionId} response={item.data} />
        ) : (
          <div
            key={`sys-${item.data.timestamp}-${item.data.type_}-${item.data.message.slice(0, 32)}`}
            className={`system-message msg-${item.data.type_.toLowerCase()}`}
          >
            <i className={`fas ${SYSTEM_ICONS[item.data.type_]}`}></i>
            <span className="system-message-text">{item.data.message}</span>
            <span className="system-message-time">{formatDateTime(item.data.timestamp)}</span>
          </div>
        ),
      )}

      {active && (
        <div className="discussion-running" role="status" aria-live="polite">
          <i className="fas fa-spinner fa-spin"></i>
          <span>Дискуссия в процессе...</span>
        </div>
      )}
    </div>
  );
};

export default DiscussionView;
