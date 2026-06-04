import React, { useEffect, useState } from 'react';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { DiscussionMeta } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import ConfirmModal from '../common/ConfirmModal';
import DiscussionCard from './DiscussionCard';
import DiscussionView from './DiscussionView';

const DiscussionHistory: React.FC = () => {
  const { showNotification } = useNotification();
  const [discussions, setDiscussions] = useState<DiscussionMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Дискуссия, ожидающая подтверждения удаления.
  const [pendingDelete, setPendingDelete] = useState<DiscussionMeta | null>(null);

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

  // Запрос удаления — открываем модалку подтверждения.
  const handleDeleteRequest = (discussionId: string) => {
    const target = discussions.find(d => d.discussion_id === discussionId) ?? null;
    setPendingDelete(target);
  };

  // Подтверждённое удаление.
  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;
    const discussionId = pendingDelete.discussion_id;
    setPendingDelete(null);

    try {
      await agentHistoryService.deleteDiscussion(discussionId);
      setDiscussions(prev => prev.filter(d => d.discussion_id !== discussionId));
      if (selectedId === discussionId) setSelectedId(null);
      showNotification('Диалог удалён', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления диалога', 'error');
    }
  };

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
          {selected?.title ?? selected?.pipeline ?? 'Без названия'}
        </h2>
        <span className="discussion-id discussion-detail-id" title={selectedId}>
          <i className="fas fa-hashtag"></i>{selectedId}
        </span>
        <DiscussionView discussionId={selectedId} />
      </div>
    );
  }

  // ── Список дискуссий ────────────────────────────────────────────────────────
  if (loading && discussions.length === 0) {
    return <div className="loading">Загрузка дискуссий...</div>;
  }

  return (
    <>
      <div className="discussions-list">
        {discussions.length === 0 ? (
          <p className="no-data">Пока нет ни одной дискуссии.</p>
        ) : (
          discussions.map(discussion => (
            <DiscussionCard
              key={discussion.discussion_id}
              discussion={discussion}
              onSelect={setSelectedId}
              onDelete={handleDeleteRequest}
            />
          ))
        )}
      </div>

      <ConfirmModal
        open={pendingDelete !== null}
        danger
        title="Удалить диалог?"
        message={
          pendingDelete
            ? `Диалог «${pendingDelete.title ?? pendingDelete.discussion_id}» будет удалён безвозвратно.`
            : undefined
        }
        confirmLabel="Удалить"
        cancelLabel="Отмена"
        onConfirm={handleConfirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </>
  );
};

export default DiscussionHistory;
