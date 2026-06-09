import React from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks';
import { POLL_INTERVAL_DISCUSSION_MS } from '../../constants';
import DiscussionInfo from './DiscussionInfo';
import DiscussionView from './DiscussionView';
import { ICONS } from '../../constants/icons';

interface Props {
  discussionId: string;
  onBack: () => void;
}

// Детальный просмотр дискуссии. Сам грузит meta и, пока дискуссия активна,
// периодически опрашивает её статус (а DiscussionView — ленту сообщений),
// чтобы показывать прогресс пайплайна в реальном времени.
const DiscussionDetail: React.FC<Props> = ({ discussionId, onBack }) => {
  const { showNotification } = useNotification();

  const { data: meta } = usePolling(
    () => agentHistoryService.getDiscussionMeta(discussionId),
    {
      intervalMs: POLL_INTERVAL_DISCUSSION_MS,
      // Продолжаем опрос только пока дискуссия активна.
      continueWhile: data => data.status === 'active',
      onError: err =>
        showNotification(err instanceof Error ? err.message : 'Ошибка загрузки дискуссии', 'error'),
      deps: [discussionId],
    },
  );

  const isActive = meta?.status === 'active';

  return (
    <div className="discussion-detail">
      <button className="detail-back-link" onClick={onBack}>
        <i className={`fas ${ICONS.back}`}></i> Назад к списку
      </button>
      <DiscussionInfo discussion={meta ?? null} discussionId={discussionId} />
      <DiscussionView discussionId={discussionId} active={isActive} />
    </div>
  );
};

export default DiscussionDetail;
