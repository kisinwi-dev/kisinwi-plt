import React from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { AgentResponse, SystemMessage, SystemMessageType } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks';
import { POLL_INTERVAL_DISCUSSION_MS } from '../../constants';
import { formatDateTime, statusClass } from '../../utils/format';
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
    // Скелетоны вместо текстовой заглушки — лента ощущается живой ещё до прихода данных.
    return (
      <div className="discussion-timeline" aria-busy="true">
        {[0, 1, 2].map(i => (
          <div key={i} className="timeline-row">
            <span className="timeline-node timeline-node--skeleton skeleton" />
            <div className="timeline-content">
              <div className="message-skeleton">
                <div className="skeleton skeleton-line skeleton-line--head" />
                <div className="skeleton skeleton-line" />
                <div className="skeleton skeleton-line skeleton-line--short" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (feed.length === 0 && !active) {
    return (
      <p className="empty-state">
        <i className="fas fa-comment-slash"></i> В этой дискуссии пока нет сообщений.
      </p>
    );
  }

  return (
    <div className="discussion-timeline">
      {feed.map(item =>
        item.kind === 'response' ? (
          <div key={item.data.response_id} className="timeline-row timeline-row--response">
            <span className={`timeline-node ${statusClass(item.data.status)}`} aria-hidden="true">
              <i className="fas fa-robot"></i>
            </span>
            <div className="timeline-content">
              <MessageBubble discussionId={discussionId} response={item.data} />
            </div>
          </div>
        ) : (
          <div
            key={`sys-${item.data.timestamp}-${item.data.type_}-${item.data.message.slice(0, 32)}`}
            className={`timeline-row timeline-row--system msg-${item.data.type_.toLowerCase()}`}
          >
            <span className="timeline-node timeline-node--system" aria-hidden="true">
              <i className={`fas ${SYSTEM_ICONS[item.data.type_]}`}></i>
            </span>
            <div className="timeline-content">
              <div className="system-message">
                <span className="system-message-text">{item.data.message}</span>
                <span className="system-message-time">{formatDateTime(item.data.timestamp)}</span>
              </div>
            </div>
          </div>
        ),
      )}

      {active && (
        <div className="timeline-row timeline-row--active" role="status" aria-live="polite">
          <span className="timeline-node timeline-node--active" aria-hidden="true">
            <span className="timeline-node-pulse"></span>
            <i className="fas fa-robot"></i>
          </span>
          <div className="timeline-content">
            <div className="discussion-running">
              <i className="fas fa-spinner fa-spin"></i>
              <span>Агенты работают...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DiscussionView;
