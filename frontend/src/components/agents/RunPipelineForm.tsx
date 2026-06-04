import React, { useEffect, useMemo, useState } from 'react';
import { datasetService } from '../../services/datasetService';
import { agentsService } from '../../services/agentsService';
import type { Dataset } from '../../types/dataset';
import { useNotification } from '../../contexts/NotificationContext';

interface Props {
  // Вызывается после успешного старта пайплайна с id созданной дискуссии.
  onStarted: (discussionId: string) => void;
}

// Разбить строку с тегами по запятой: trim, без пустых.
const parseTags = (raw: string): string[] =>
  raw.split(',').map(t => t.trim()).filter(Boolean);

// Разбить многострочный текст по строкам: trim, без пустых.
const parseLines = (raw: string): string[] =>
  raw.split('\n').map(t => t.trim()).filter(Boolean);

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
  const [tags, setTags] = useState('');
  const [deniedHypotheses, setDeniedHypotheses] = useState('');
  const [maxIter, setMaxIter] = useState(2);

  const [submitting, setSubmitting] = useState(false);

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
        denied_hypotheses_info: parseLines(deniedHypotheses),
        max_iter: maxIter,
        title: title.trim() || undefined,
        tags: parseTags(tags),
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
        <h3>Данные</h3>
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
        </div>
      </div>

      <div className="form-section">
        <h3>Параметры модели</h3>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="run-model">Имя модели <span className="required-star">*</span></label>
            <input
              id="run-model"
              type="text"
              placeholder="например: resnet50"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={submitting}
            />
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
            <textarea
              id="run-denied"
              placeholder="По одной на строку (необязательно)"
              value={deniedHypotheses}
              onChange={(e) => setDeniedHypotheses(e.target.value)}
              rows={2}
              disabled={submitting}
            />
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>Оформление запуска</h3>
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
          <div className="form-field">
            <label htmlFor="run-tags">Теги</label>
            <input
              id="run-tags"
              type="text"
              placeholder="через запятую"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              disabled={submitting}
            />
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
