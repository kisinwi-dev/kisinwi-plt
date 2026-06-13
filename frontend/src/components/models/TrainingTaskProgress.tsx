import React, { useEffect, useState } from 'react';
import { taskerService } from '../../services/taskerService';
import { useNotification } from '../../contexts/NotificationContext';
import { usePolling } from '../../hooks/usePolling';
import ConfirmModal from '../common/ConfirmModal';
import type { TrainingTask } from '../../types/tasks';
import { formatDateTime, formatElapsed, parseBackendDate } from '../../utils/format';
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

  const percent = Math.min(100, Math.max(0, task.percentages));
  // В очереди прогресса ещё нет — показываем indeterminate-полосу вместо пустой.
  const waiting = task.status === 'waiting';

  const startedMs = parseBackendDate(task.started_at);
  const completedMs = parseBackendDate(task.completed_at);
  const elapsedMs = startedMs === null
    ? null
    : active ? now - startedMs
    : completedMs !== null ? completedMs - startedMs
    : null;

  return (
    <section className="detail-section">
      <h3 className="detail-section-title"><i className={`fas ${ICONS.task}`}></i> Процесс обучения</h3>

      <div className="training-task">
        <div className="training-task-header">
          <span className={statusBadgeClass(task.status)}>
            {active && <><i className={`fas ${ICONS.loading} fa-spin`}></i>{' '}</>}
            {taskStatusLabel(task)}
          </span>
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
              className={`progress-bar${waiting ? ' progress-bar--indeterminate' : ''}`}
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              {...(waiting ? { 'aria-valuetext': 'В очереди' } : { 'aria-valuenow': percent })}
            >
              {waiting
                ? <div className="progress-bar-runner" />
                : <div className="progress-bar-fill" style={{ width: `${percent}%` }} />}
            </div>
            {!waiting && <span className="training-task-percent">{percent}%</span>}
          </div>
        )}

        {task.status_info && <p className="training-task-status-info">{task.status_info}</p>}

        {task.status === 'failed' && task.error_message && (
          <p className="training-task-error">
            <i className={`fas ${ICONS.error}`}></i>
            <span>{task.error_message}</span>
          </p>
        )}

        <div className="training-task-meta">
          {task.started_at && (
            <span className="training-task-meta-item">
              <i className={`fas ${ICONS.duration}`}></i> Старт:
              <span className="training-task-meta-value">{formatDateTime(task.started_at)}</span>
            </span>
          )}
          {task.completed_at && (
            <span className="training-task-meta-item">
              <i className={`fas ${ICONS.dateFinished}`}></i> Завершение:
              <span className="training-task-meta-value">{formatDateTime(task.completed_at)}</span>
            </span>
          )}
          {elapsedMs !== null && (
            <span className="training-task-meta-item">
              <i className={`fas ${ICONS.elapsed}`}></i> Длительность:
              <span className="training-task-meta-value">{formatElapsed(elapsedMs)}</span>
            </span>
          )}
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
