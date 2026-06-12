import React, { useState } from 'react';
import { taskerService } from '../../services/taskerService';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks/usePolling';
import ConfirmModal from '../common/ConfirmModal';
import type { TrainingTask } from '../../types/tasks';
import { formatDateTime } from '../../utils/format';
import { ICONS } from '../../constants/icons';
import { statusBadgeClass, POLL_INTERVAL_TASK_MS } from '../../constants';

// Статусы, в которых задача ещё живая: опрашиваем и даём отменить.
const ACTIVE_TASK_STATUSES = ['waiting', 'running'];

// Человекочитаемые статусы задачи; для неизвестных — описание из tasker.
const TASK_STATUS_LABELS: Record<string, string> = {
  waiting: 'В очереди',
  running: 'Обучается',
  completed: 'Завершена',
  failed: 'Ошибка',
  cancelled: 'Отменена',
};

const taskStatusLabel = (task: TrainingTask): string =>
  TASK_STATUS_LABELS[task.status] ?? task.status_description ?? task.status;

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

  const active = ACTIVE_TASK_STATUSES.includes(task.status);
  const percent = Math.min(100, Math.max(0, task.percentages));

  return (
    <section className="detail-section">
      <h3 className="detail-section-title"><i className={`fas ${ICONS.task}`}></i> Процесс обучения</h3>

      <div className="training-task">
        <div className="training-task-header">
          <span className={statusBadgeClass(task.status)}>
            {active && <><i className={`fas ${ICONS.loading} fa-spin`}></i>{' '}</>}
            {taskStatusLabel(task)}
          </span>
          {task.status_info && <span className="training-task-info">{task.status_info}</span>}
          {active && (
            <button
              className="button danger small training-task-cancel"
              onClick={() => setPendingCancel(true)}
              disabled={cancelling}
            >
              <i className={`fas ${ICONS.cancelled}`}></i>
              {cancelling ? 'Отмена…' : 'Отменить обучение'}
            </button>
          )}
        </div>

        {active && (
          <div className="training-task-progress">
            <div
              className="progress-bar"
              role="progressbar"
              aria-valuenow={percent}
              aria-valuemin={0}
              aria-valuemax={100}
            >
              <div className="progress-bar-fill" style={{ width: `${percent}%` }} />
            </div>
            <span className="training-task-percent">{percent}%</span>
          </div>
        )}

        {task.status === 'failed' && task.error_message && (
          <p className="training-task-error">
            <i className={`fas ${ICONS.error}`}></i> {task.error_message}
          </p>
        )}

        <div className="training-task-meta">
          {task.started_at && <span><i className={`fas ${ICONS.duration}`}></i> Старт: {formatDateTime(task.started_at)}</span>}
          {task.completed_at && <span>Завершение: {formatDateTime(task.completed_at)}</span>}
        </div>
      </div>

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
