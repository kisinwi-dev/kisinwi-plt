import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { AgentResponse, AgentStatus, Tool } from '../../types/agentHistory';
import { formatDateTime, formatDuration, statusClass } from '../../utils/format';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';

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

// Отображение одного вызова инструмента: шапка-список + раскрываемые подробности.
const ToolItem: React.FC<{ tool: Tool }> = ({ tool }) => {
  const [expanded, setExpanded] = useState(false);

  const hasDetails =
    Boolean(tool.message) || tool.input_args != null || tool.output != null || Boolean(tool.error_traceback);

  return (
    <div className={`tool-item ${expanded ? 'expanded' : ''}`}>
      <div
        className={`tool-header ${hasDetails ? 'clickable' : ''}`}
        {...(hasDetails ? getDisclosureProps(expanded, () => setExpanded(prev => !prev)) : {})}
      >
        <span className="tool-name">
          {hasDetails && <CollapseChevron open={expanded} />}
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
  const [tools, setTools] = useState<Tool[] | 'loading' | 'error' | null>(null);
  const [collapsed, setCollapsed] = useState(true);

  const handleToggleTools = async (e: React.MouseEvent) => {
    e.stopPropagation();
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
    <div className={`message-bubble ${statusClass(response.status)}${collapsed ? ' message-bubble--collapsed' : ''}`}>
      <div
        className="message-header message-header--clickable"
        {...getDisclosureProps(!collapsed, () => setCollapsed(prev => !prev))}
      >
        <span className="message-header-left">
          <CollapseChevron open={!collapsed} />
          <span className="message-role"><i className="fas fa-robot"></i> {response.agent_role}</span>
        </span>
        <div className="message-header-right">
          <span className={`status-badge ${statusClass(response.status)}`}>{STATUS_LABELS[response.status]}</span>
        </div>
      </div>

      {!collapsed && (
        <>
          <div className="message-meta">
            {response.model && <span><i className="fas fa-microchip"></i> {response.model}</span>}
            {response.task_name && <span><i className="fas fa-list-check"></i> {response.task_name}</span>}
            {response.iteration != null && <span><i className="fas fa-rotate"></i> итерация {response.iteration}</span>}
            {response.duration_ms != null && <span><i className="fas fa-clock"></i> {formatDuration(response.duration_ms)}</span>}
          </div>

          <div className="message-text markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{response.text}</ReactMarkdown>
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
        </>
      )}
    </div>
  );
};

export default MessageBubble;
