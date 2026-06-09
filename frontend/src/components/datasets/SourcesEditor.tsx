import React, { useState } from 'react';
import Select from '../common/Select';
import './SourcesEditor.css';
import { ICONS } from '../../constants/icons';
import type { SourceItem } from '../../types/dataset';

const SOURCE_TYPE_OPTIONS = [
  { value: 'kaggle', label: '📊 Kaggle' },
  { value: 'url', label: '🌐 URL' },
  { value: 'huggingface', label: '🤗 Hugging Face' },
  { value: 'other', label: '📁 Другой' },
];

const TYPE_LABEL: Record<SourceItem['type'], string> = {
  kaggle: '📊 Kaggle',
  url: '🌐 URL',
  huggingface: '🤗 Hugging Face',
  other: '📁 Другой',
};

const EMPTY_DRAFT = (): SourceItem => ({ type: 'kaggle', url: null, description: '' });

interface SourcesEditorProps {
  sources: SourceItem[];
  loading: boolean;
  onChange: (sources: SourceItem[]) => void;
}

// Редактор источников по паттерну draft → chip: поля черновика (тип + URL + описание)
// добавляются кнопкой в список, каждый добавленный источник — строка с крестиком.
// Единый блок для формы создания датасета и формы добавления версии.
const SourcesEditor: React.FC<SourcesEditorProps> = ({ sources, loading, onChange }) => {
  const [draft, setDraft] = useState<SourceItem>(EMPTY_DRAFT());

  const canAdd = Boolean(draft.url?.trim() || draft.description.trim());

  const addSource = () => {
    if (!canAdd) return;
    onChange([...sources, { ...draft, url: draft.url?.trim() || null, description: draft.description.trim() }]);
    setDraft(EMPTY_DRAFT());
  };

  const removeSource = (index: number) => {
    onChange(sources.filter((_, i) => i !== index));
  };

  return (
    <div className="sources-editor">
      <div className="source-draft">
        <div className="source-draft-type">
          <Select
            ariaLabel="Тип источника"
            value={draft.type}
            options={SOURCE_TYPE_OPTIONS}
            onChange={(v) => setDraft((d) => ({ ...d, type: v as SourceItem['type'] }))}
          />
        </div>
        <input
          type="url"
          autoComplete="off"
          placeholder="https://..."
          value={draft.url ?? ''}
          onChange={(e) => setDraft((d) => ({ ...d, url: e.target.value || null }))}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); addSource(); }
          }}
          disabled={loading}
        />
        <input
          type="text"
          autoComplete="off"
          placeholder="Описание, например: оригинальный источник"
          value={draft.description}
          onChange={(e) => setDraft((d) => ({ ...d, description: e.target.value }))}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); addSource(); }
          }}
          disabled={loading}
        />
        <button
          type="button"
          className="button secondary small"
          onClick={addSource}
          disabled={loading || !canAdd}
        >
          <i className={`fas ${ICONS.add}`}></i> Добавить
        </button>
      </div>

      {sources.length > 0 && (
        <ul className="source-list">
          {sources.map((source, index) => (
            <li key={index} className="source-row">
              <span className="source-row-type">{TYPE_LABEL[source.type]}</span>
              {source.url && <span className="source-row-url">{source.url}</span>}
              {source.description && <span className="source-row-desc">{source.description}</span>}
              <button
                type="button"
                className="source-row-remove"
                onClick={() => removeSource(index)}
                disabled={loading}
                aria-label="Удалить источник"
                title="Удалить источник"
              >
                <i className={`fas ${ICONS.close}`}></i>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SourcesEditor;
