import React, { useEffect, useState } from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { AgentResponse, SystemMessage, SystemMessageType } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import { formatDateTime } from '../../utils/format';
import MessageBubble from './MessageBubble';

interface Props {
  discussionId: string;
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

const DiscussionView: React.FC<Props> = ({ discussionId }) => {
  const { showNotification } = useNotification();
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchFeed = async () => {
      try {
        setLoading(true);
        const [responses, systemMessages] = await Promise.all([
          agentHistoryService.getResponses(discussionId),
          agentHistoryService.getSystemMessages(discussionId),
        ]);

        const items: FeedItem[] = [
          ...responses.map((data): FeedItem => ({ kind: 'response', timestamp: data.timestamp, data })),
          ...systemMessages.map((data): FeedItem => ({ kind: 'system', timestamp: data.timestamp, data })),
        ];
        items.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        setFeed(items);
      } catch (err) {
        showNotification(err instanceof Error ? err.message : 'Ошибка загрузки диалога', 'error');
      } finally {
        setLoading(false);
      }
    };
    fetchFeed();
  }, [discussionId, showNotification]);

  if (loading) {
    return <div className="loading">Загрузка диалога...</div>;
  }

  if (feed.length === 0) {
    return <p className="no-data">В этой дискуссии пока нет сообщений.</p>;
  }

  return (
    <div className="discussion-feed">
      {feed.map((item, index) =>
        item.kind === 'response' ? (
          <MessageBubble key={item.data.response_id} discussionId={discussionId} response={item.data} />
        ) : (
          <div key={`sys-${index}`} className={`system-message msg-${item.data.type_.toLowerCase()}`}>
            <i className={`fas ${SYSTEM_ICONS[item.data.type_]}`}></i>
            <span className="system-message-text">{item.data.message}</span>
            <span className="system-message-time">{formatDateTime(item.data.timestamp)}</span>
          </div>
        ),
      )}
    </div>
  );
};

export default DiscussionView;
