import React from 'react';
import type { DiscussionMeta } from '../../types/agentHistory';
import { DISCUSSION_STATUS_LABELS, getDiscussionTitle, getDiscussionAgents, getPipelineLabel } from '../../types/agentHistory';
import type { TrainingState, TrainingSummary } from './discussionFeed';
import { formatDateTime } from '../../utils/format';
import { ICONS } from '../../constants/icons';
import { useCopyToClipboard } from '../../hooks';
import { Tooltip } from '../common/Tooltip';

interface Props {
  discussion: DiscussionMeta | null;
  discussionId: string;
  // Роль агента, который сейчас работает (его карточка пульсирует). null — никто.
  activeAgentRole?: string | null;
  // Сводка обучения для карточки «Сервис обучения» в секции исполнителей.
  training?: TrainingSummary;
  actions?: React.ReactNode;
}

// Иконка статус-строки карточки сервиса обучения по состоянию.
const TRAINING_ICON: Record<TrainingState, string> = {
  idle: ICONS.duration,
  running: ICONS.loading,
  completed: ICONS.success,
  failed: ICONS.error,
  cancelled: ICONS.cancelled,
};

// Подпись статус-строки с учётом нескольких последовательных обучений (total).
const trainingStatusLabel = ({ state, total }: TrainingSummary): string => {
  switch (state) {
    case 'running':
      return total > 1 ? `Идёт обучение (модель ${total})` : 'Идёт обучение';
    case 'completed':
      return total > 1 ? `Обучено моделей: ${total}` : 'Обучение завершено';
    case 'failed':
      return 'Ошибка обучения';
    case 'cancelled':
      return 'Обучение отменено';
    default:
      return 'Не запущено';
  }
};

const DiscussionInfo: React.FC<Props> = ({
  discussion,
  discussionId,
  activeAgentRole = null,
  training = { state: 'idle', total: 0, currentName: null },
  actions,
}) => {
  const copyToClipboard = useCopyToClipboard();
  const title = discussion ? getDiscussionTitle(discussion) : 'Без названия';
  const agents = discussion ? getDiscussionAgents(discussion) : [];

  return (
    <>
      <header className="detail-header">
        <div className="detail-heading">
          <div className="detail-title">
            <h1>{title}</h1>
            {discussion && (
              <span className={`status-badge status-${discussion.status}`}>
                <span className={`status-dot ${discussion.status === 'active' ? 'status-dot--pulse' : ''}`} aria-hidden="true" />
                {DISCUSSION_STATUS_LABELS[discussion.status]}
              </span>
            )}
          </div>
          <Tooltip content="Нажмите, чтобы скопировать ID">
            <span className="discussion-id" onClick={() => copyToClipboard(discussionId)}>
              <i className={`fas ${ICONS.id}`}></i>{discussionId}
              <i className={`fas ${ICONS.copy} discussion-id-copy-icon`}></i>
            </span>
          </Tooltip>
        </div>
        {actions && <div className="detail-header-actions">{actions}</div>}
      </header>

      {discussion && (
        <>
          <section className="detail-section">
            <h3 className="detail-section-title"><i className={`fas ${ICONS.info}`}></i> Общая информация</h3>
            <div className="detail-fields">
              {discussion.pipeline && (
                <div className="detail-field">
                  <span className="detail-label"><i className={`fas ${ICONS.pipeline}`}></i> Пайплайн</span>
                  <span>{getPipelineLabel(discussion.pipeline)}</span>
                </div>
              )}
              <div className="detail-field">
                <span className="detail-label"><i className={`fas ${ICONS.dateCreated}`}></i> Создано</span>
                <span>{formatDateTime(discussion.created_at)}</span>
              </div>
              {discussion.finished_at && (
                <div className="detail-field">
                  <span className="detail-label"><i className={`fas ${ICONS.dateFinished}`}></i> Завершено</span>
                  <span>{formatDateTime(discussion.finished_at)}</span>
                </div>
              )}
              {discussion.responses_count != null && (
                <div className="detail-field">
                  <span className="detail-label"><i className={`fas ${ICONS.discussion}`}></i> Вызовы агентов</span>
                  <span>{discussion.responses_count}</span>
                </div>
              )}
              {discussion.tool_calls_count != null && discussion.tool_calls_count > 0 && (
                <div className="detail-field">
                  <span className="detail-label"><i className={`fas ${ICONS.tools}`}></i> Вызовов инструментов</span>
                  <span>{discussion.tool_calls_count}</span>
                </div>
              )}
              {discussion.tags.length > 0 && (
                <div className="detail-field detail-field--full">
                  <span className="detail-label"><i className={`fas ${ICONS.tag}`}></i> Теги</span>
                  <div className="tag-list">
                    {discussion.tags.map(tag => (
                      <span key={tag} className="tag">{tag}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="detail-section">
            <h3 className="detail-section-title">
              <i className={`fas ${ICONS.agentsGroup}`}></i> Исполнители ({agents.length + 1})
            </h3>
            <div className="discussion-agents">
              {agents.map(agent => (
                <div
                  key={agent.role}
                  className={`agent-card ${agent.role === activeAgentRole ? 'agent-card--active' : ''}`}
                >
                  <span className="agent-card-avatar">
                    <i className={`fas ${ICONS.agent}`}></i>
                  </span>
                  <div className="agent-card-body">
                    <Tooltip content={agent.role} className="agent-card-role">{agent.role}</Tooltip>
                    {agent.models.length > 0 && (
                      <div className="agent-card-models">
                        {agent.models.map(model => (
                          <Tooltip key={model} content={model} className="agent-model-badge">
                            <i className={`fas ${ICONS.agentModel}`}></i>
                            <span className="agent-model-text">{model}</span>
                          </Tooltip>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Постоянная карточка сервиса обучения: пульсирует, пока идёт обучение. */}
              <div
                className={`agent-card agent-card--training ${training.state === 'running' ? 'agent-card--active' : ''}`}
              >
                <span className="agent-card-avatar agent-card-avatar--training">
                  <i className={`fas ${ICONS.model}`}></i>
                </span>
                <div className="agent-card-body">
                  <span className="agent-card-role">Сервис обучения</span>
                  <div className="agent-card-models">
                    <span className={`training-status training-status--${training.state}`}>
                      <i className={`fas ${TRAINING_ICON[training.state]} ${training.state === 'running' ? 'fa-spin' : ''}`} aria-hidden="true"></i>
                      <span className="training-status-text">{trainingStatusLabel(training)}</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </>
      )}
    </>
  );
};

export default DiscussionInfo;
