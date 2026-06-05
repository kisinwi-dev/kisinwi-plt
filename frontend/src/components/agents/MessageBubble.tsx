import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { AgentResponse, AgentStatus, Tool, ToolStatus } from '../../types/agentHistory';
import { formatDateTime } from '../../utils/format';

interface Props {
  discussionId: string;
  response: AgentResponse;
}

// Подписи статусов запуска агента / инструмента.
const STATUS_LABELS: Record<AgentStatus, string> = {
  'IN PROGRESS': 'В процессе',
  SUCCEED: 'Успешно',
  ERROR: 'Ошибка',
};

// Класс статуса для бейджа (IN PROGRESS -> in-progress).
const statusClass = (status: AgentStatus | ToolStatus): string =>
  `status-${status.toLowerCase().replace(/\s+/g, '-')}`;

// Форматирование длительности в мс/с.
const formatDuration = (ms: number): string =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)} с` : `${ms.toFixed(0)} мс`;

// Отображение одного вызова инструмента: шапка-список + раскрываемые подробности.
const ToolItem: React.FC<{ tool: Tool }> = ({ tool }) => {
  const [expanded, setExpanded] = useState(false);

  const hasDetails =
    Boolean(tool.message) || tool.input_args != null || tool.output != null || Boolean(tool.error_traceback);

  return (
    <div className={`tool-item ${expanded ? 'expanded' : ''}`}>
      <div
        className={`tool-header ${hasDetails ? 'clickable' : ''}`}
        onClick={hasDetails ? () => setExpanded(prev => !prev) : undefined}
        role={hasDetails ? 'button' : undefined}
        tabIndex={hasDetails ? 0 : undefined}
        onKeyDown={hasDetails ? (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpanded(prev => !prev); }
        } : undefined}
      >
        <span className="tool-name">
          {hasDetails && <i className={`fas tool-chevron ${expanded ? 'fa-chevron-down' : 'fa-chevron-right'}`}></i>}
          <i className="fas fa-wrench"></i> {tool.name}
        </span>
        <div className="tool-header-right">
          {tool.duration_ms != null && <span className="tool-duration"><i className="fas fa-clock"></i> {formatDuration(tool.duration_ms)}</span>}
          <span className={`status-badge ${statusClass(tool.status)}`}>{STATUS_LABELS[tool.status]}</span>
        </div>
      </div>

      {expanded && hasDetails && (
        <div className="tool-details">
          {tool.message && <p className="tool-message">{tool.message}</p>}
          {tool.input_args != null && (
            <div className="tool-block">
              <span className="tool-block-label">Аргументы</span>
              <pre>{JSON.stringify(tool.input_args, null, 2)}</pre>
            </div>
          )}
          {tool.output != null && (
            <div className="tool-block">
              <span className="tool-block-label">Результат</span>
              <pre>{typeof tool.output === 'string' ? tool.output : JSON.stringify(tool.output, null, 2)}</pre>
            </div>
          )}
          {tool.error_traceback && (
            <div className="tool-block tool-block--error">
              <span className="tool-block-label">Traceback</span>
              <pre>{tool.error_traceback}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const MessageBubble: React.FC<Props> = ({ discussionId, response }) => {
  // null — инструменты не загружались; иначе toggle с состоянием загрузки.
  const [tools, setTools] = useState<Tool[] | 'loading' | 'error' | null>(null);

  const handleToggleTools = async () => {
    if (tools !== null) {
      setTools(null);
      return;
    }
    setTools('loading');
    try {
      const data = await agentHistoryService.getToolsByResponse(discussionId, response.response_id);
      setTools(data);
    } catch {
      setTools('error');
    }
  };

  return (
    <div className={`message-bubble ${statusClass(response.status)}`}>
      <div className="message-header">
        <span className="message-role"><i className="fas fa-robot"></i> {response.agent_role}</span>
        <span className={`status-badge ${statusClass(response.status)}`}>{STATUS_LABELS[response.status]}</span>
      </div>

      <div className="message-meta">
        {response.model && <span><i className="fas fa-microchip"></i> {response.model}</span>}
        {response.task_name && <span><i className="fas fa-list-check"></i> {response.task_name}</span>}
        {response.iteration != null && <span><i className="fas fa-rotate"></i> итерация {response.iteration}</span>}
        {response.duration_ms != null && <span><i className="fas fa-clock"></i> {formatDuration(response.duration_ms)}</span>}
      </div>

      <div className="message-text markdown-body">
        <ReactMarkdown>{response.text}</ReactMarkdown>
      </div>

      <div className="message-footer">
        <span className="message-time">{formatDateTime(response.timestamp)}</span>
        <button className="tools-toggle" onClick={handleToggleTools}>
          <i className={`fas ${tools !== null ? 'fa-chevron-up' : 'fa-wrench'}`}></i>
          {tools !== null ? ' Скрыть инструменты' : ' Инструменты'}
        </button>
      </div>

      {tools === 'loading' && <p className="tools-status">Загрузка инструментов...</p>}
      {tools === 'error' && <p className="tools-status tools-status--error">Не удалось загрузить инструменты</p>}
      {Array.isArray(tools) && (
        tools.length === 0
          ? <p className="tools-status">Инструменты не вызывались</p>
          : <div className="tools-list">{tools.map(tool => <ToolItem key={tool.id} tool={tool} />)}</div>
      )}
    </div>
  );
};

export default MessageBubble;
