import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { agentHistoryService } from '../../services/agentHistoryService';
import { taskerService } from '../../services/taskerService';
import type { AgentResponse, SystemMessage, SystemMessageType } from '../../types/agentHistory';
import type { TrainingTask } from '../../types/tasks';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks';
import { POLL_INTERVAL_DISCUSSION_MS } from '../../constants';
import { formatDateTime, parseBackendDate, statusClass } from '../../utils/format';
import MessageBubble from './MessageBubble';
import TrainingTaskCard from '../models/TrainingTaskCard';
import { ICONS } from '../../constants/icons';

interface Props {
  discussionId: string;
  // Если дискуссия активна — лента периодически обновляется (polling прогресса).
  active?: boolean;
}

// Элемент единой ленты: ответ агента, системное сообщение или этап обучения.
type FeedItem =
  | { kind: 'response'; timestamp: string; data: AgentResponse }
  | { kind: 'system'; timestamp: string; data: SystemMessage }
  | { kind: 'training'; timestamp: string; data: TrainingTask };

// Иконка системного сообщения по типу.
const SYSTEM_ICONS: Record<SystemMessageType, string> = {
  INFO: ICONS.info,
  WARNING: ICONS.warning,
  ERROR: ICONS.error,
};

const DiscussionView: React.FC<Props> = ({ discussionId, active = false }) => {
  const { showNotification } = useNotification();
  const navigate = useNavigate();

  const { data, loading } = usePolling<FeedItem[]>(
    async () => {
      // Задачи обучения дискуссии берём из tasker по discussion_id — это даёт
      // живой прогресс этапа обучения и ссылку на модель прямо в ленте истории.
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
      items.sort(
        (a, b) => (parseBackendDate(a.timestamp) ?? 0) - (parseBackendDate(b.timestamp) ?? 0),
      );
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

  // Живой таймер длительности для активного этапа обучения: пока дискуссия
  // активна, «сейчас» тикает раз в секунду (карточка обучения считает elapsed).
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);

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
        <i className={`fas ${ICONS.noMessages}`}></i> В этой дискуссии пока нет сообщений.
      </p>
    );
  }

  return (
    <div className="discussion-timeline">
      {feed.map(item => {
        if (item.kind === 'response') {
          return (
            <div key={item.data.response_id} className="timeline-row timeline-row--response">
              <span className={`timeline-node ${statusClass(item.data.status)}`} aria-hidden="true">
                <i className={`fas ${ICONS.agent}`}></i>
              </span>
              <div className="timeline-content">
                <MessageBubble discussionId={discussionId} response={item.data} />
              </div>
            </div>
          );
        }
        if (item.kind === 'training') {
          const task = item.data;
          return (
            <div key={`train-${task.id}`} className="timeline-row timeline-row--training">
              <span className="timeline-node timeline-node--training" aria-hidden="true">
                <i className={`fas ${ICONS.task}`}></i>
              </span>
              <div className="timeline-content">
                <div className="training-stage">
                  <div className="training-stage-header">
                    <span className="training-stage-title">
                      <i className={`fas ${ICONS.model}`}></i> {task.name}
                    </span>
                    <button
                      className="button small training-stage-link"
                      onClick={() => navigate(`/models/${task.model_id}`)}
                    >
                      <i className={`fas ${ICONS.external}`}></i> Открыть страницу модели
                    </button>
                  </div>
                  <TrainingTaskCard task={task} now={now} />
                </div>
              </div>
            </div>
          );
        }
        return (
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
        );
      })}

      {active && (
        <div className="timeline-row timeline-row--active" role="status" aria-live="polite">
          <span className="timeline-node timeline-node--active" aria-hidden="true">
            <span className="timeline-node-pulse"></span>
            <i className={`fas ${ICONS.agent}`}></i>
          </span>
          <div className="timeline-content">
            <div className="discussion-running">
              <i className={`fas ${ICONS.loading} fa-spin`}></i>
              <span>Агенты работают...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DiscussionView;
