import React, { useEffect, useMemo, useState } from 'react';
import { datasetService } from '../../services/datasetService';
import { agentsService } from '../../services/agentsService';
import type { Dataset } from '../../types/dataset';
import { useNotification } from '../../contexts/NotificationContext';
import Combobox from '../common/Combobox';
import ChipListEditor from '../common/ChipListEditor';

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
  // Запрещённые гипотезы добавляются по одной в список.
  const [deniedList, setDeniedList] = useState<string[]>([]);
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
    <form className="run-pipeline-form" onSubmit={handleSubmit} autoComplete="off">
      <h2>Запуск пайплайна разработки</h2>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className="fas fa-database"></i> Данные и модель</h3>
          <p className="form-section-hint">Что обучаем и на каких данных.</p>
        </div>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="run-dataset">Имя датасета <span className="required-star">*</span></label>
            <Combobox
              id="run-dataset"
              icon="fas fa-database"
              placeholder="например: CIFAR-10"
              value={datasetName}
              options={datasets.map(d => d.name)}
              onChange={(v) => { setDatasetName(v); setVersionName(''); }}
              disabled={submitting}
            />
          </div>
          <div className="form-field">
            <label htmlFor="run-version">Имя версии <span className="required-star">*</span></label>
            <Combobox
              id="run-version"
              icon="fas fa-code-branch"
              placeholder="например: v1.0"
              value={versionName}
              options={matchedDataset?.versions.map(v => v.name) ?? []}
              onChange={setVersionName}
              disabled={submitting}
            />
          </div>
          <div className="form-field">
            <label htmlFor="run-model">Имя модели <span className="required-star">*</span></label>
            <input
              id="run-model"
              type="text"
              autoComplete="off"
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
              className="no-spinner"
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
              autoComplete="off"
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
              autoComplete="off"
              placeholder="Технические ограничения, например: только CPU, ≤100 МБ"
              value={deploymentConstraints}
              onChange={(e) => setDeploymentConstraints(e.target.value)}
              rows={3}
              disabled={submitting}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="run-denied">Запрещённые гипотезы и практики</label>
            <ChipListEditor
              id="run-denied"
              variant="row"
              items={deniedList}
              onChange={setDeniedList}
              placeholder="Например: не использовать аугментацию поворотом"
              disabled={submitting}
            />
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
              autoComplete="off"
              placeholder="Необязательно — иначе сгенерируется автоматически"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={submitting}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="run-tags">Теги</label>
            <ChipListEditor
              id="run-tags"
              variant="chip"
              items={tagList}
              onChange={setTagList}
              placeholder="Например: эксперимент"
              suggestions={SUGGESTED_TAGS}
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
