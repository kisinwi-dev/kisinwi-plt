import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { agentHistoryService } from '../../services/agentHistoryService';
import type { AgentResponse, AgentStatus, Tool } from '../../types/agentHistory';
import type { AgentTokenMetrics } from '../../services/metricsService';
import { formatDateTime, formatDuration, formatCompact, statusClass } from '../../utils/format';
import { CollapseChevron, getDisclosureProps } from '../common/Collapse';
import { ICONS } from '../../constants/icons';

interface Props {
  discussionId: string;
  response: AgentResponse;
  // Токены, потраченные этим агентом (из metrics-сервиса); может ещё не быть.
  tokens?: AgentTokenMetrics;
}

// Подписи статусов запуска агента / инструмента.
const STATUS_LABELS: Record<AgentStatus, string> = {
  'IN PROGRESS': 'В процессе',
  SUCCEED: 'Успешно',
  ERROR: 'Ошибка',
};

// Превью текста ответа для свёрнутой карточки: грубая зачистка markdown до плоской строки.
const toPreview = (text: string, max = 140): string => {
  const flat = text
    .replace(/```[\s\S]*?```/g, ' ')           // блоки кода
    .replace(/[#>*_`~-]+/g, ' ')               // markdown-разметка
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')   // ссылки → текст
    .replace(/\s+/g, ' ')
    .trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
};

// Отображение одного вызова инструмента: шапка-список + раскрываемые подробности.
// maxDuration — наибольшая длительность среди инструментов ответа (для ширины latency-bar).
const ToolItem: React.FC<{ tool: Tool; maxDuration: number }> = ({ tool, maxDuration }) => {
  const [expanded, setExpanded] = useState(false);

  const hasDetails =
    Boolean(tool.message) || tool.input_args != null || tool.output != null || Boolean(tool.error_traceback);

  // Относительная ширина бара длительности (видно «тяжёлые» вызовы).
  const barWidth =
    tool.duration_ms != null && maxDuration > 0
      ? `${Math.max(4, (tool.duration_ms / maxDuration) * 100)}%`
      : null;

  return (
    <div className={`tool-item ${statusClass(tool.status)} ${expanded ? 'expanded' : ''}`}>
      <div
        className={`tool-header ${hasDetails ? 'clickable' : ''}`}
        {...(hasDetails ? getDisclosureProps(expanded, () => setExpanded(prev => !prev)) : {})}
      >
        <span className="tool-name">
          {hasDetails && <CollapseChevron open={expanded} />}
          <i className={`fas ${ICONS.tools}`}></i> {tool.name}
        </span>
        <div className="tool-header-right">
          {tool.duration_ms != null && (
            <span className="tool-duration">
              {barWidth && (
                <span className="tool-latency-bar" aria-hidden="true">
                  <span className="tool-latency-bar-fill" style={{ width: barWidth }} />
                </span>
              )}
              <i className={`fas ${ICONS.duration}`}></i> {formatDuration(tool.duration_ms)}
            </span>
          )}
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

const MessageBubble: React.FC<Props> = ({ discussionId, response, tokens }) => {
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
          <span className="message-role">{response.agent_role}</span>
        </span>
        <div className="message-header-right">
          {response.duration_ms != null && (
            <span className="message-header-duration"><i className={`fas ${ICONS.duration}`}></i> {formatDuration(response.duration_ms)}</span>
          )}
          <span className={`status-badge ${statusClass(response.status)}`}>{STATUS_LABELS[response.status]}</span>
        </div>
      </div>

      {collapsed && response.text.trim() && (
        <p className="message-preview">{toPreview(response.text)}</p>
      )}

      {!collapsed && (
        <>
          <div className="message-meta">
            {response.model && <span><i className={`fas ${ICONS.agentModel}`}></i> {response.model}</span>}
            {response.task_name && <span><i className={`fas ${ICONS.task}`}></i> {response.task_name}</span>}
            {response.iteration != null && <span><i className={`fas ${ICONS.iteration}`}></i> итерация {response.iteration}</span>}
            {response.duration_ms != null && <span><i className={`fas ${ICONS.duration}`}></i> {formatDuration(response.duration_ms)}</span>}
            {tokens?.total_tokens != null && tokens.total_tokens > 0 && (
              <span title={`${tokens.total_tokens.toLocaleString('ru-RU')} токенов (prompt ${(tokens.prompt_tokens ?? 0).toLocaleString('ru-RU')} / completion ${(tokens.completion_tokens ?? 0).toLocaleString('ru-RU')})`}>
                <i className={`fas ${ICONS.tokens}`}></i> {formatCompact(tokens.total_tokens)} токенов
              </span>
            )}
          </div>

          <div className="message-text markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{response.text}</ReactMarkdown>
          </div>

          <div className="message-footer">
            <span className="message-time">{formatDateTime(response.timestamp)}</span>
            <button className="tools-toggle" onClick={handleToggleTools}>
              <i className={`fas ${tools !== null ? ICONS.collapse : ICONS.tools}`}></i>
              {tools !== null ? ' Скрыть инструменты' : ' Инструменты'}
            </button>
          </div>

          {tools === 'loading' && <p className="tools-status">Загрузка инструментов...</p>}
          {tools === 'error' && <p className="tools-status tools-status--error">Не удалось загрузить инструменты</p>}
          {Array.isArray(tools) && (
            tools.length === 0
              ? <p className="tools-status">Инструменты не вызывались</p>
              : (() => {
                  const maxDuration = Math.max(0, ...tools.map(t => t.duration_ms ?? 0));
                  return (
                    <div className="tools-list">
                      {tools.map(tool => <ToolItem key={tool.id} tool={tool} maxDuration={maxDuration} />)}
                    </div>
                  );
                })()
          )}
        </>
      )}
    </div>
  );
};

export default MessageBubble;
