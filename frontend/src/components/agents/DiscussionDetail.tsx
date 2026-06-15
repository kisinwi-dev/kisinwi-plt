import React, { useMemo, useState } from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import { agentsService } from '../../services/agentsService';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks';
import { POLL_INTERVAL_DISCUSSION_MS } from '../../constants';
import DiscussionInfo from './DiscussionInfo';
import DiscussionView from './DiscussionView';
import {
  loadDiscussionFeed,
  deriveActiveAgentRole,
  deriveTrainingSummary,
  type FeedItem,
} from './discussionFeed';
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

  // Лента грузится здесь (а не в DiscussionView): её данные нужны и внизу (сама лента),
  // и сверху (индикаторы активного агента и состояния обучения в шапке).
  const { data: feedData, loading: feedLoading } = usePolling<FeedItem[]>(
    () => loadDiscussionFeed(discussionId),
    {
      intervalMs: POLL_INTERVAL_DISCUSSION_MS,
      // Ленту грузим хотя бы раз; повторяем опрос только пока дискуссия активна.
      continueWhile: () => isActive,
      onError: err =>
        showNotification(err instanceof Error ? err.message : 'Ошибка загрузки диалога', 'error'),
      // isActive в deps: когда meta догрузилась и статус стал active — цикл опроса перезапускается.
      deps: [discussionId, isActive],
    },
  );

  const feed = useMemo(() => feedData ?? [], [feedData]);
  // Активный агент значим только пока дискуссия активна.
  const activeAgentRole = isActive ? deriveActiveAgentRole(feed) : null;
  const training = deriveTrainingSummary(feed);

  const [stopping, setStopping] = useState(false);

  const handleStop = async () => {
    setStopping(true);
    try {
      await agentsService.stopPipeline(discussionId);
      showNotification('Остановка агентов запрошена', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Не удалось остановить агентов', 'error');
    } finally {
      setStopping(false);
    }
  };

  return (
    <div className="discussion-detail">
      <button className="detail-back-link" onClick={onBack}>
        <i className={`fas ${ICONS.back}`}></i> Назад к списку
      </button>
      <DiscussionInfo
        discussion={meta ?? null}
        discussionId={discussionId}
        activeAgentRole={activeAgentRole}
        training={training}
        actions={isActive ? (
          <button className="button danger small" onClick={handleStop} disabled={stopping}>
            <i className={`fas ${stopping ? `${ICONS.loading} fa-spin` : ICONS.cancelled}`}></i>
            {stopping ? 'Остановка…' : 'Остановить агентов'}
          </button>
        ) : undefined}
      />
      <DiscussionView
        discussionId={discussionId}
        feed={feed}
        loading={feedLoading}
        active={isActive}
      />
    </div>
  );
};

export default DiscussionDetail;
