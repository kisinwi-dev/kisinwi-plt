import React, { useEffect, useState } from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { DiscussionMeta } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import DiscussionCard from './DiscussionCard';
import DiscussionView from './DiscussionView';

const DiscussionHistory: React.FC = () => {
  const { showNotification } = useNotification();
  const [discussions, setDiscussions] = useState<DiscussionMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchDiscussions = async () => {
      try {
        setLoading(true);
        const data = await agentHistoryService.getDiscussions();
        setDiscussions(data);
      } catch (err) {
        showNotification(err instanceof Error ? err.message : 'Ошибка загрузки дискуссий', 'error');
      } finally {
        setLoading(false);
      }
    };
    fetchDiscussions();
  }, [showNotification]);

  const selected = selectedId
    ? discussions.find(d => d.discussion_id === selectedId)
    : null;

  // ── Детальный просмотр выбранной дискуссии ──────────────────────────────────
  if (selectedId) {
    return (
      <div className="discussion-detail">
        <button className="button secondary small back-button" onClick={() => setSelectedId(null)}>
          <i className="fas fa-arrow-left"></i> Назад к списку
        </button>
        <h2 className="discussion-detail-title">
          {selected?.title ?? selectedId}
        </h2>
        <DiscussionView discussionId={selectedId} />
      </div>
    );
  }

  // ── Список дискуссий ────────────────────────────────────────────────────────
  if (loading && discussions.length === 0) {
    return <div className="loading">Загрузка дискуссий...</div>;
  }

  return (
    <div className="discussions-list">
      {discussions.length === 0 ? (
        <p className="no-data">Пока нет ни одной дискуссии.</p>
      ) : (
        discussions.map(discussion => (
          <DiscussionCard
            key={discussion.discussion_id}
            discussion={discussion}
            onSelect={setSelectedId}
          />
        ))
      )}
    </div>
  );
};

export default DiscussionHistory;
