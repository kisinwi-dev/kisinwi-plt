import React, { useEffect, useMemo, useState } from 'react';
import { datasetService } from '../../services/datasetService';
import { agentsService } from '../../services/agentsService';
import type { Dataset } from '../../types/dataset';
import { useNotification } from '../../contexts/NotificationContext';

interface Props {
  // Вызывается после успешного старта пайплайна с id созданной дискуссии.
  onStarted: (discussionId: string) => void;
}

// Заглушка предустановленных тегов. Позже будем подтягивать жёстко заданные
// теги из сервиса истории агентов вместо этого хардкода.
const SUGGESTED_TAGS = ['эксперимент', 'baseline', 'продакшн-кандидат', 'быстрый прогон'];

const RunPipelineForm: React.FC<Props> = ({ onStarted }) => {
  const { showNotification } = useNotification();

  // Датасеты грузим один раз — для резолва имя→id и подсказок.
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // Поля формы.
  const [datasetName, setDatasetName] = useState('');
  const [versionName, setVersionName] = useState('');
  const [modelName, setModelName] = useState('');
  const [businessRequirements, setBusinessRequirements] = useState('');
  const [deploymentConstraints, setDeploymentConstraints] = useState('');
  const [title, setTitle] = useState('');
  // Теги добавляются по одному в список.
  const [tagList, setTagList] = useState<string[]>([]);
  const [tagDraft, setTagDraft] = useState('');
  // Запрещённые гипотезы добавляются по одной в список.
  const [deniedList, setDeniedList] = useState<string[]>([]);
  const [deniedDraft, setDeniedDraft] = useState('');
  const [maxIter, setMaxIter] = useState(2);

  const [submitting, setSubmitting] = useState(false);

  // Добавить тег (из черновика или переданный из предложенных), без дублей.
  const addTag = (raw?: string) => {
    const value = (raw ?? tagDraft).trim();
    if (!value) return;
    setTagList(prev => (prev.includes(value) ? prev : [...prev, value]));
    setTagDraft('');
  };
  const removeTag = (index: number) => {
    setTagList(prev => prev.filter((_, i) => i !== index));
  };

  // Добавить запрещённую гипотезу из черновика в список (без дублей).
  const addDenied = () => {
    const value = deniedDraft.trim();
    if (!value) return;
    setDeniedList(prev => (prev.includes(value) ? prev : [...prev, value]));
    setDeniedDraft('');
  };
  const removeDenied = (index: number) => {
    setDeniedList(prev => prev.filter((_, i) => i !== index));
  };

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const data = await datasetService.getDatasets();
        setDatasets(data);
      } catch (err) {
        showNotification(
          err instanceof Error ? err.message : 'Ошибка загрузки датасетов',
          'error',
        );
      }
    };
    fetchDatasets();
  }, [showNotification]);

  // Датасет, совпавший по имени с введённым (для подсказок версий).
  const matchedDataset = useMemo(
    () => datasets.find(d => d.name.trim() === datasetName.trim()) ?? null,
    [datasets, datasetName],
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;

    // Резолв имя→id датасета и версии.
    const dataset = datasets.find(d => d.name.trim() === datasetName.trim());
    if (!dataset) {
      showNotification(`Датасет «${datasetName}» не найден`, 'error');
      return;
    }
    const version = dataset.versions.find(v => v.name.trim() === versionName.trim());
    if (!version) {
      showNotification(`Версия «${versionName}» не найдена в датасете «${dataset.name}»`, 'error');
      return;
    }

    // Валидация обязательных текстовых полей.
    if (!modelName.trim()) {
      showNotification('Укажите имя модели', 'error');
      return;
    }
    if (!businessRequirements.trim()) {
      showNotification('Укажите бизнес-требования', 'error');
      return;
    }
    if (!deploymentConstraints.trim()) {
      showNotification('Укажите ограничения прода', 'error');
      return;
    }

    setSubmitting(true);
    try {
      const result = await agentsService.startDevelopment({
        dataset_id: dataset.id,
        version_id: version.id,
        model_name: modelName.trim(),
        business_requirements: businessRequirements.trim(),
        deployment_constraints: deploymentConstraints.trim(),
        denied_hypotheses_info: deniedList,
        max_iter: maxIter,
        title: title.trim() || undefined,
        tags: tagList,
      });
      showNotification('Пайплайн запущен', 'success');
      onStarted(result.discussion_id);
    } catch (err) {
      showNotification(
        err instanceof Error ? err.message : 'Не удалось запустить пайплайн',
        'error',
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="run-pipeline-form" onSubmit={handleSubmit}>
      <h2>Запуск пайплайна разработки</h2>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-database"></i> Данные и модель</h3>
          <p className="form-section-hint">Что обучаем и на каких данных.</p>
        </div>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="run-dataset">Имя датасета <span className="required-star">*</span></label>
            <input
              id="run-dataset"
              type="text"
              list="run-datasets-list"
              placeholder="например: CIFAR-10"
              value={datasetName}
              onChange={(e) => { setDatasetName(e.target.value); setVersionName(''); }}
              disabled={submitting}
            />
            <datalist id="run-datasets-list">
              {datasets.map(d => <option key={d.id} value={d.name} />)}
            </datalist>
          </div>
          <div className="form-field">
            <label htmlFor="run-version">Имя версии <span className="required-star">*</span></label>
            <input
              id="run-version"
              type="text"
              list="run-versions-list"
              placeholder="например: v1.0"
              value={versionName}
              onChange={(e) => setVersionName(e.target.value)}
              disabled={submitting}
            />
            <datalist id="run-versions-list">
              {matchedDataset?.versions.map(v => <option key={v.id} value={v.name} />)}
            </datalist>
          </div>
          <div className="form-field">
            <label htmlFor="run-model">Имя модели <span className="required-star">*</span></label>
            <input
              id="run-model"
              type="text"
              placeholder="придумайте сами, например: my-model"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={submitting}
            />
            <span className="field-hint">Произвольное имя — под ним сохранится обученная модель.</span>
          </div>
          <div className="form-field">
            <label htmlFor="run-max-iter">Попыток обучения</label>
            <input
              id="run-max-iter"
              type="number"
              min={1}
              value={maxIter}
              onChange={(e) => setMaxIter(Math.max(1, Number(e.target.value) || 1))}
              disabled={submitting}
            />
          </div>
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-circle-info"></i> Контекст для агентов</h3>
          <p className="form-section-hint">Эту информацию агенты используют при подборе решения.</p>
        </div>
        <div className="form-grid">
          <div className="form-field full-width">
            <label htmlFor="run-business">Бизнес-требования <span className="required-star">*</span></label>
            <textarea
              id="run-business"
              placeholder="Что нужно бизнесу от модели"
              value={businessRequirements}
              onChange={(e) => setBusinessRequirements(e.target.value)}
              rows={3}
              disabled={submitting}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="run-deployment">Ограничения прода <span className="required-star">*</span></label>
            <textarea
              id="run-deployment"
              placeholder="Технические ограничения, например: только CPU, ≤100 МБ"
              value={deploymentConstraints}
              onChange={(e) => setDeploymentConstraints(e.target.value)}
              rows={3}
              disabled={submitting}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="run-denied">Запрещённые гипотезы и практики</label>
            <div className="denied-input-row">
              <input
                id="run-denied"
                type="text"
                placeholder="Например: не использовать аугментацию поворотом"
                value={deniedDraft}
                onChange={(e) => setDeniedDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { e.preventDefault(); addDenied(); }
                }}
                disabled={submitting}
              />
              <button
                type="button"
                className="button secondary small"
                onClick={addDenied}
                disabled={submitting || !deniedDraft.trim()}
              >
                <i className="fas fa-plus"></i> Добавить
              </button>
            </div>
            {deniedList.length > 0 && (
              <ul className="denied-list">
                {deniedList.map((item, index) => (
                  <li key={item} className="denied-item">
                    <span className="denied-item-text">{item}</span>
                    <button
                      type="button"
                      className="denied-item-remove"
                      onClick={() => removeDenied(index)}
                      disabled={submitting}
                      aria-label="Удалить"
                      title="Удалить"
                    >
                      <i className="fas fa-xmark"></i>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-tag"></i> Параметры запуска</h3>
          <p className="form-section-hint">Как этот запуск будет назван в истории.</p>
        </div>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="run-title">Название запуска</label>
            <input
              id="run-title"
              type="text"
              placeholder="Необязательно — иначе сгенерируется автоматически"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={submitting}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="run-tags">Теги</label>
            <div className="denied-input-row">
              <input
                id="run-tags"
                type="text"
                placeholder="Например: эксперимент"
                value={tagDraft}
                onChange={(e) => setTagDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { e.preventDefault(); addTag(); }
                }}
                disabled={submitting}
              />
              <button
                type="button"
                className="button secondary small"
                onClick={() => addTag()}
                disabled={submitting || !tagDraft.trim()}
              >
                <i className="fas fa-plus"></i> Добавить
              </button>
            </div>
            {SUGGESTED_TAGS.filter(t => !tagList.includes(t)).length > 0 && (
              <div className="tag-suggestions">
                <span className="tag-suggestions-label">Предложенные:</span>
                {SUGGESTED_TAGS.filter(t => !tagList.includes(t)).map(t => (
                  <button
                    key={t}
                    type="button"
                    className="tag-suggestion"
                    onClick={() => addTag(t)}
                    disabled={submitting}
                  >
                    <i className="fas fa-plus"></i> {t}
                  </button>
                ))}
              </div>
            )}
            {tagList.length > 0 && (
              <ul className="tag-list">
                {tagList.map((tag, index) => (
                  <li key={tag} className="tag-chip">
                    <span className="tag-chip-text">{tag}</span>
                    <button
                      type="button"
                      className="tag-chip-remove"
                      onClick={() => removeTag(index)}
                      disabled={submitting}
                      aria-label="Удалить тег"
                      title="Удалить"
                    >
                      <i className="fas fa-xmark"></i>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <div className="run-pipeline-actions">
        <button type="submit" className="button" disabled={submitting}>
          {submitting ? (
            <><i className="fas fa-spinner fa-spin"></i> Запуск...</>
          ) : (
            <><i className="fas fa-play"></i> Запустить</>
          )}
        </button>
      </div>
    </form>
  );
};

export default RunPipelineForm;
