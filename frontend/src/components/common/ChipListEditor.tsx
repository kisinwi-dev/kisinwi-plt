import React, { useState } from 'react';
import { ICONS } from '../../constants/icons';
import './ChipListEditor.css';

interface Props {
  /** Текущий список строк. */
  items: string[];
  /** Вызывается с новым списком при добавлении/удалении. */
  onChange: (items: string[]) => void;
  placeholder?: string;
  /** Подпись на кнопке добавления (по умолчанию «Добавить»). */
  addLabel?: string;
  /** Предложенные значения — рендерятся как chip-подсказки над списком. */
  suggestions?: string[];
  /** Подпись к блоку предложенных. */
  suggestionsLabel?: string;
  /** 'chip' — компактные пилюли с переносом; 'row' — список строк во всю ширину. */
  variant?: 'chip' | 'row';
  disabled?: boolean;
  id?: string;
}

/**
 * Редактор списка строк по паттерну «draft-input → Enter/кнопка → chip с крестиком».
 * Владеет состоянием черновика; наружу отдаёт только готовый список через onChange.
 * Trim + дедупликация при добавлении.
 */
const ChipListEditor: React.FC<Props> = ({
  items,
  onChange,
  placeholder,
  addLabel = 'Добавить',
  suggestions,
  suggestionsLabel = 'Предложенные:',
  variant = 'chip',
  disabled,
  id,
}) => {
  const [draft, setDraft] = useState('');

  const add = (raw?: string) => {
    const value = (raw ?? draft).trim();
    if (!value) return;
    if (!items.includes(value)) onChange([...items, value]);
    setDraft('');
  };

  const remove = (index: number) => {
    onChange(items.filter((_, i) => i !== index));
  };

  const visibleSuggestions = suggestions?.filter((s) => !items.includes(s)) ?? [];

  return (
    <div className="chip-editor">
      <div className="chip-input-row">
        <input
          id={id}
          type="text"
          autoComplete="off"
          placeholder={placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              add();
            }
          }}
          disabled={disabled}
        />
        <button
          type="button"
          className="button secondary small"
          onClick={() => add()}
          disabled={disabled || !draft.trim()}
        >
          <i className={`fas ${ICONS.add}`}></i> {addLabel}
        </button>
      </div>

      {visibleSuggestions.length > 0 && (
        <div className="chip-suggestions">
          <span className="chip-suggestions-label">{suggestionsLabel}</span>
          {visibleSuggestions.map((s) => (
            <button
              key={s}
              type="button"
              className="chip-suggestion"
              onClick={() => add(s)}
              disabled={disabled}
            >
              <i className={`fas ${ICONS.add}`}></i> {s}
            </button>
          ))}
        </div>
      )}

      {items.length > 0 && (
        <ul className={`chip-list chip-list--${variant}`}>
          {items.map((item, index) => (
            <li key={item} className="chip">
              <span className="chip-text">{item}</span>
              <button
                type="button"
                className="chip-remove"
                onClick={() => remove(index)}
                disabled={disabled}
                aria-label="Удалить"
                title="Удалить"
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

export default ChipListEditor;
