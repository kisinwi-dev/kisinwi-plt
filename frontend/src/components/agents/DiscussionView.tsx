import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { SystemMessageType } from '../../types/agentHistory';
import { formatDateTime, statusClass } from '../../utils/format';
import { prefersReducedMotion } from '../../utils/motion';
import MessageBubble from './MessageBubble';
import TrainingTaskCard from '../models/TrainingTaskCard';
import type { FeedItem } from './discussionFeed';
import type { AgentTokenMetrics } from '../../services/metricsService';
import { ICONS } from '../../constants/icons';

interface Props {
  discussionId: string;
  // Лента дискуссии (грузится в DiscussionDetail — единый источник правды).
  feed: FeedItem[];
  // true, пока идёт первичная загрузка ленты (показываем скелетоны).
  loading?: boolean;
  // Если дискуссия активна — показываем индикатор «Агенты работают...».
  active?: boolean;
  // Токены по response_id (из metrics-сервиса) — показываются на карточке ответа.
  tokensByResponse?: Map<string, AgentTokenMetrics>;
}

// Иконка системного сообщения по типу.
const SYSTEM_ICONS: Record<SystemMessageType, string> = {
  INFO: ICONS.info,
  WARNING: ICONS.warning,
  ERROR: ICONS.error,
};

// Насколько близко к низу страницы считаем, что пользователь «внизу», px.
const SCROLL_BOTTOM_THRESHOLD = 120;

const DiscussionView: React.FC<Props> = ({ discussionId, feed, loading = false, active = false, tokensByResponse }) => {
  const navigate = useNavigate();

  // Живой таймер длительности для активного этапа обучения: пока дискуссия
  // активна, «сейчас» тикает раз в секунду (карточка обучения считает elapsed).
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);

  // Автоскролл: если пользователь у низа страницы, новая запись в ленте плавно
  // спускает вьюпорт вниз, а не наращивает высоту под текущей позицией.
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const nearBottomRef = useRef(true);
  const prevLenRef = useRef(feed.length);

  useEffect(() => {
    const onScroll = () => {
      const scrolledToBottom =
        window.innerHeight + window.scrollY >=
        document.documentElement.scrollHeight - SCROLL_BOTTOM_THRESHOLD;
      nearBottomRef.current = scrolledToBottom;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const grew = feed.length > prevLenRef.current;
    prevLenRef.current = feed.length;
    // Спускаем вниз только для живой дискуссии (новые записи приходят, пока active),
    // чтобы при открытии завершённой не дёргать пользователя к низу.
    if (active && grew && nearBottomRef.current) {
      bottomRef.current?.scrollIntoView({
        behavior: prefersReducedMotion() ? 'auto' : 'smooth',
        block: 'end',
      });
    }
  }, [feed.length, active]);

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
                <MessageBubble discussionId={discussionId} response={item.data} tokens={tokensByResponse?.get(item.data.response_id)} />
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

      {/* Якорь автоскролла: к нему подъезжаем при появлении новой записи. */}
      <div ref={bottomRef} aria-hidden="true" />
    </div>
  );
};

export default DiscussionView;
