import React, { useEffect, useState } from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { DiscussionMeta } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import DiscussionInfo from './DiscussionInfo';
import DiscussionView from './DiscussionView';

interface Props {
  discussionId: string;
  onBack: () => void;
}

// Интервал опроса прогресса активной дискуссии.
const POLL_INTERVAL_MS = 3000;

// Детальный просмотр дискуссии. Сам грузит meta и, пока дискуссия активна,
// периодически опрашивает её статус (а DiscussionView — ленту сообщений),
// чтобы показывать прогресс пайплайна в реальном времени.
const DiscussionDetail: React.FC<Props> = ({ discussionId, onBack }) => {
  const { showNotification } = useNotification();
  const [meta, setMeta] = useState<DiscussionMeta | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const data = await agentHistoryService.getDiscussionMeta(discussionId);
        if (cancelled) return;
        setMeta(data);
        // Продолжаем опрос только пока дискуссия активна.
        if (data.status === 'active') {
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        showNotification(
          err instanceof Error ? err.message : 'Ошибка загрузки дискуссии',
          'error',
        );
      }
    };

    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [discussionId, showNotification]);

  const isActive = meta?.status === 'active';

  return (
    <div className="discussion-detail">
      <button className="button secondary small back-button" onClick={onBack}>
        <i className="fas fa-arrow-left"></i> Назад к списку
      </button>
      <DiscussionInfo discussion={meta} discussionId={discussionId} />
      <DiscussionView discussionId={discussionId} active={isActive} />
    </div>
  );
};

export default DiscussionDetail;
