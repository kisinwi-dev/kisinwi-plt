import React from 'react';
import type { TrainingTask } from '../../types/tasks';
import { formatDateTime, formatElapsed, parseBackendDate } from '../../utils/format';
import { ICONS } from '../../constants/icons';
import { statusBadgeClass } from '../../constants';

// Статусы, в которых задача ещё живая: опрашиваем и даём отменить.
export const ACTIVE_TASK_STATUSES = ['waiting', 'running'];

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

interface TrainingTaskCardProps {
  task: TrainingTask;
  // «Сейчас» для живого таймера длительности (тикает у родителя раз в секунду).
  now: number;
  // Если передан — у активной задачи показываем кнопку отмены.
  onCancel?: () => void;
  cancelling?: boolean;
}

/**
 * Презентация хода обучения: статус, прогресс-бар, статусная строка, ошибка и
 * длительность. Без опроса tasker и без модалок — только отображение готовой
 * задачи. Используется и на странице модели (TrainingTaskProgress), и в ленте
 * истории дискуссии (DiscussionView).
 */
const TrainingTaskCard: React.FC<TrainingTaskCardProps> = ({ task, now, onCancel, cancelling }) => {
  const active = ACTIVE_TASK_STATUSES.includes(task.status);
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
    <div className="training-task">
      <div className="training-task-header">
        <span className={statusBadgeClass(task.status)}>
          {active && <><i className={`fas ${ICONS.loading} fa-spin`}></i>{' '}</>}
          {taskStatusLabel(task)}
        </span>
        {active && onCancel && (
          <button
            className="button danger small training-task-cancel"
            onClick={onCancel}
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
  );
};

export default TrainingTaskCard;
