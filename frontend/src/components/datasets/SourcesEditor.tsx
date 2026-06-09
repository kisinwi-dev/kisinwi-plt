import React from 'react';
import type { SourceItem } from '../../types/dataset';

interface SourcesEditorProps {
  sources: SourceItem[];
  loading: boolean;
  onAdd: () => void;
  onRemove: (index: number) => void;
  onChange: (index: number, field: keyof SourceItem, value: string) => void;
}

// Редактор списка источников данных версии: тип, URL, описание + добавление/удаление.
// Единый блок для формы создания датасета и формы добавления версии.
const SourcesEditor: React.FC<SourcesEditorProps> = ({ sources, loading, onAdd, onRemove, onChange }) => (
  <div className="sources-container">
    {sources.map((source, index) => (
      <div key={index} className="source-card">
        <div className="source-header">
          <select
            value={source.type}
            onChange={(e) => onChange(index, 'type', e.target.value)}
            className="source-type-select"
            disabled={loading}
          >
            <option value="kaggle">📊 Kaggle</option>
            <option value="url">🌐 URL</option>
            <option value="huggingface">🤗 Hugging Face</option>
            <option value="other">📁 Другой</option>
          </select>
          {sources.length > 1 && (
            <button
              type="button"
              className="source-remove-btn"
              onClick={() => onRemove(index)}
              title="Удалить источник"
              disabled={loading}
            >
              <i className="fas fa-trash-alt"></i>
            </button>
          )}
        </div>
        <div className="source-fields">
          <div className="form-field">
            <label>URL источника</label>
            <input
              type="url"
              placeholder="https://..."
              value={source.url ?? ''}
              onChange={(e) => onChange(index, 'url', e.target.value)}
              className="source-input"
              disabled={loading}
            />
          </div>
          <div className="form-field">
            <label>Описание</label>
            <input
              type="text"
              placeholder="например: оригинальный источник"
              value={source.description}
              onChange={(e) => onChange(index, 'description', e.target.value)}
              className="source-input"
              disabled={loading}
            />
          </div>
        </div>
      </div>
    ))}
    <button type="button" className="add-source-btn" onClick={onAdd} disabled={loading}>
      <i className="fas fa-plus-circle"></i> Добавить ещё источник
    </button>
  </div>
);

export default SourcesEditor;
