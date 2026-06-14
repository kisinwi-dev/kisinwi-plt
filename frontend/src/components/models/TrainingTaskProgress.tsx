import React, { useEffect, useState } from 'react';
import { taskerService } from '../../services/taskerService';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks/usePolling';
import ConfirmModal from '../common/ConfirmModal';
import TrainingTaskCard, { ACTIVE_TASK_STATUSES } from './TrainingTaskCard';
import type { TrainingTask } from '../../types/tasks';
import { ICONS } from '../../constants/icons';
import { POLL_INTERVAL_TASK_MS } from '../../constants';

interface TrainingTaskProgressProps {
  modelId: string;
}

/**
 * Виджет хода обучения на странице модели: прогресс задачи из tasker
 * (проценты, статусная строка) с возможностью отмены. Самодостаточен:
 * сам опрашивает tasker и не рендерится, если показывать нечего
 * (задач нет или последняя завершилась успешно — метрики уже ниже).
 */
const TrainingTaskProgress: React.FC<TrainingTaskProgressProps> = ({ modelId }) => {
  const { showNotification } = useNotification();
  const [pendingCancel, setPendingCancel] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  // Берём последнюю задачу модели: при переобучении задач может быть несколько.
  const { data: task, refetch } = usePolling<TrainingTask | null>(
    async () => {
      const { tasks } = await taskerService.getTasks({ model_id: modelId });
      if (tasks.length === 0) return null;
      return [...tasks].sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
    },
    {
      intervalMs: POLL_INTERVAL_TASK_MS,
      // Финальный статус — опрос больше не нужен; tasker недоступен — не
      // долбим его, страница модели живёт без виджета.
      continueWhile: (t) => t !== null && ACTIVE_TASK_STATUSES.includes(t.status),
      deps: [modelId],
    },
  );

  const active = !!task && task.model_id === modelId && ACTIVE_TASK_STATUSES.includes(task.status);

  // Живой таймер длительности: пока задача активна, «сейчас» тикает раз в секунду.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);

  const handleCancel = async () => {
    if (!task) return;
    setPendingCancel(false);
    try {
      setCancelling(true);
      await taskerService.cancelTask(task.id);
      showNotification('Обучение будет остановлено на границе эпохи', 'success');
      refetch();
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось отменить задачу', 'error');
    } finally {
      setCancelling(false);
    }
  };

  // Нечего показывать: задач нет либо последняя завершилась успешно.
  // Проверка model_id отсекает устаревшие данные прошлой версии: при переключении
  // версии usePolling отдаёт прежний результат до первого ответа нового опроса.
  if (!task || task.model_id !== modelId || task.status === 'completed') return null;

  return (
    <section className="detail-section">
      <h3 className="detail-section-title"><i className={`fas ${ICONS.task}`}></i> Процесс обучения</h3>

      <TrainingTaskCard
        task={task}
        now={now}
        onCancel={() => setPendingCancel(true)}
        cancelling={cancelling}
      />

      <ConfirmModal
        open={pendingCancel}
        danger
        title="Отменить обучение?"
        message={`Задача «${task.name}» будет отменена. Воркер остановит обучение на границе эпохи.`}
        confirmLabel="Отменить обучение"
        cancelLabel="Назад"
        onConfirm={handleCancel}
        onCancel={() => setPendingCancel(false)}
      />
    </section>
  );
};

export default TrainingTaskProgress;
