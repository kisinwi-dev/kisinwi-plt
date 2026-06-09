import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { DiscussionMeta } from '../../types/agentHistory';
import { useNotification } from '../../contexts/NotificationContext';
import ConfirmModal from '../common/ConfirmModal';
import DiscussionCard from './DiscussionCard';

const DiscussionHistory: React.FC = () => {
  const { showNotification } = useNotification();
  const navigate = useNavigate();

  const [discussions, setDiscussions] = useState<DiscussionMeta[]>([]);
  const [loading, setLoading] = useState(false);
  // Дискуссия, ожидающая подтверждения удаления.
  const [pendingDelete, setPendingDelete] = useState<DiscussionMeta | null>(null);

  // Открыть дискуссию — переход на отдельную страницу детального просмотра.
  const openDiscussion = (discussionId: string) => {
    navigate(`/agents/discussion/${discussionId}`);
  };

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
      showNotification('Диалог удалён', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления диалога', 'error');
    }
  };

  // ── Список дискуссий ────────────────────────────────────────────────────────
  if (loading && discussions.length === 0) {
    return <div className="loading-state">Загрузка дискуссий...</div>;
  }

  return (
    <>
      <div className="discussions-list">
        {discussions.length === 0 ? (
          <p className="empty-state">Пока нет ни одной дискуссии.</p>
        ) : (
          discussions.map(discussion => (
            <DiscussionCard
              key={discussion.discussion_id}
              discussion={discussion}
              onSelect={openDiscussion}
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
