import React, { useEffect, useMemo, useState } from 'react';
import { datasetService } from '../../services/datasetService';
import { agentsService } from '../../services/agentsService';
import { mlModelsService } from '../../services/mlModelsService';
import type { Dataset } from '../../types/dataset';
import type { MLModel } from '../../types/mlModels';
import type { LlmModelInfo } from '../../types/llm';
import { useNotification } from '../../contexts/NotificationContext';
import Combobox from '../common/Combobox';
import ChipListEditor from '../common/ChipListEditor';
import { ICONS } from '../../constants/icons';

interface Props {
  // Вызывается после успешного старта пайплайна с id созданной дискуссии.
  onStarted: (discussionId: string) => void;
}

// Заглушка предустановленных тегов. Позже будем подтягивать жёстко заданные
// теги из сервиса истории агентов вместо этого хардкода.
const SUGGESTED_TAGS = ['эксперимент', 'baseline', 'продакшн-кандидат', 'быстрый прогон'];

type Workflow = 'development' | 'quick';
type ModelMode = 'new' | 'existing';

const MODEL_MODES: { id: ModelMode; label: string; hint: string }[] = [
  {
    id: 'new',
    label: 'Создать новую',
    hint: 'Модель появится в реестре под указанным именем.',
  },
  {
    id: 'existing',
    label: 'Продолжить существующую',
    hint: 'Агенты обучат новые версии выбранной модели с учётом её истории.',
  },
];

const WORKFLOWS: { id: Workflow; icon: string; label: string; hint: string }[] = [
  {
    id: 'development',
    icon: ICONS.pipeline,
    label: 'Полный цикл',
    hint: 'Все агенты: анализ датасета, гипотезы, обучение с итерациями, дебаг и отчёт.',
  },
  {
    id: 'quick',
    icon: ICONS.quickRun,
    label: 'Быстрый прогон',
    hint: 'Только ML-инженер и аналитик метрик: конфигурация, обучение, разбор результата.',
  },
];

const RunPipelineForm: React.FC<Props> = ({ onStarted }) => {
  const { showNotification } = useNotification();

  // Датасеты грузим один раз — для резолва имя→id и подсказок.
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  // Существующие модели — для выбора в режиме «продолжить существующую».
  const [models, setModels] = useState<MLModel[]>([]);
  // Каталог LLM-моделей агентов (подсказки) и модель по умолчанию.
  const [llmModels, setLlmModels] = useState<LlmModelInfo[]>([]);
  const [llmDefaultModel, setLlmDefaultModel] = useState('');
  // Модель на этот запуск. Пусто — используется модель по умолчанию.
  const [llmModel, setLlmModel] = useState('');

  // Поля формы.
  const [workflow, setWorkflow] = useState<Workflow>('development');
  const [datasetName, setDatasetName] = useState('');
  const [versionName, setVersionName] = useState('');
  const [modelMode, setModelMode] = useState<ModelMode>('new');
  const [modelName, setModelName] = useState('');
  const [existingModelName, setExistingModelName] = useState('');
  const [businessRequirements, setBusinessRequirements] = useState('');
  const [deploymentConstraints, setDeploymentConstraints] = useState('');
  const [title, setTitle] = useState('');
  // Теги добавляются по одному в список.
  const [tagList, setTagList] = useState<string[]>([]);
  // Запрещённые гипотезы добавляются по одной в список.
  const [deniedList, setDeniedList] = useState<string[]>([]);
  const [maxIter, setMaxIter] = useState(0);
  // Игнорировать вердикт аналитика данных, если он забраковал датасет.
  const [skipDatasetCheck, setSkipDatasetCheck] = useState(false);

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

    const fetchModels = async () => {
      try {
        const data = await mlModelsService.getModels();
        setModels(data.models);
      } catch (err) {
        showNotification(
          err instanceof Error ? err.message : 'Ошибка загрузки моделей',
          'error',
        );
      }
    };
    fetchModels();

    const fetchLlmSettings = async () => {
      try {
        const settings = await agentsService.getLlmSettings();
        setLlmModels(settings.available);
        // Поле оставляем пустым: пусто = модель по умолчанию (current_model).
        setLlmDefaultModel(settings.current_model);
      } catch (err) {
        showNotification(
          err instanceof Error ? err.message : 'Ошибка загрузки списка LLM-моделей',
          'error',
        );
      }
    };
    fetchLlmSettings();
  }, [showNotification]);

  // Подсказки для Combobox — id моделей из каталога. Ввод остаётся свободным:
  // можно указать любую модель OpenRouter вручную.
  const llmModelOptions = useMemo<string[]>(
    () => llmModels.map(m => m.id),
    [llmModels],
  );

  // Датасет, совпавший по имени с введённым (для подсказок версий).
  const matchedDataset = useMemo(
    () => datasets.find(d => d.name.trim() === datasetName.trim()) ?? null,
    [datasets, datasetName],
  );

  // Существующая модель, совпавшая по имени с выбранной (для hint и резолва id).
  const matchedModel = useMemo(
    () => models.find(m => m.name.trim() === existingModelName.trim()) ?? null,
    [models, existingModelName],
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

    // Валидация модели: новая — по имени, существующая — резолв имя→id.
    let modelPayload: { model_name?: string; model_id?: string };
    if (modelMode === 'existing') {
      const model = models.find(m => m.name.trim() === existingModelName.trim());
      if (!model) {
        showNotification(`Модель «${existingModelName}» не найдена`, 'error');
        return;
      }
      modelPayload = { model_id: model.id, model_name: model.name };
    } else {
      if (!modelName.trim()) {
        showNotification('Укажите имя модели', 'error');
        return;
      }
      modelPayload = { model_name: modelName.trim() };
    }
    setSubmitting(true);
    try {
      const commonPayload = {
        dataset_id: dataset.id,
        version_id: version.id,
        ...modelPayload,
        business_requirements: businessRequirements.trim() || undefined,
        deployment_constraints: deploymentConstraints.trim() || undefined,
        title: title.trim() || undefined,
        tags: tagList,
        llm_model: llmModel.trim() || undefined,
      };
      const result = workflow === 'quick'
        ? await agentsService.startQuickTraining(commonPayload)
        : await agentsService.startDevelopment({
            ...commonPayload,
            denied_hypotheses_info: deniedList,
            max_iter: maxIter,
            skip_dataset_check: skipDatasetCheck,
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
          <h3><i className={`fas ${ICONS.pipeline}`}></i> Воркфлоу</h3>
          <p className="form-section-hint">Какие агенты участвуют в запуске.</p>
        </div>
        <div className="workflow-selector" role="radiogroup" aria-label="Воркфлоу">
          {WORKFLOWS.map(w => (
            <button
              key={w.id}
              type="button"
              role="radio"
              aria-checked={workflow === w.id}
              className={`workflow-option${workflow === w.id ? ' active' : ''}`}
              onClick={() => setWorkflow(w.id)}
              disabled={submitting}
            >
              <i className={`fas ${w.icon}`}></i>
              <span className="workflow-option-label">{w.label}</span>
              <span className="workflow-option-hint">{w.hint}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className={`fas ${ICONS.agentModel}`}></i> Модель агентов (LLM)</h3>
          <p className="form-section-hint">Какая LLM управляет работой агентов в этом запуске. Можно переопределить глобальную настройку.</p>
        </div>
        <div className="form-grid">
          <div className="form-field full-width">
            <label htmlFor="run-llm-model">LLM-модель</label>
            <Combobox
              id="run-llm-model"
              icon={`fas ${ICONS.agentModel}`}
              placeholder={llmDefaultModel ? `По умолчанию: ${llmDefaultModel}` : 'Модель по умолчанию'}
              value={llmModel}
              options={llmModelOptions}
              onChange={setLlmModel}
              disabled={submitting}
            />
            <span className="field-hint">Можно выбрать из каталога или ввести любую модель OpenRouter вручную. Пусто — используется модель по умолчанию.</span>
          </div>
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className={`fas ${ICONS.dataset}`}></i> Данные и модель</h3>
          <p className="form-section-hint">Что обучаем и на каких данных.</p>
        </div>
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="run-dataset">Имя датасета <span className="required-star">*</span></label>
            <Combobox
              id="run-dataset"
              icon={`fas ${ICONS.dataset}`}
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
              icon={`fas ${ICONS.version}`}
              placeholder="например: v1.0"
              value={versionName}
              options={matchedDataset?.versions.map(v => v.name) ?? []}
              onChange={setVersionName}
              disabled={submitting}
            />
          </div>
          {workflow === 'development' && (
            <label className={`form-field full-width skip-check-toggle${skipDatasetCheck ? ' active' : ''}`}>
              <span className="skip-check-text">
                <span className="skip-check-label">
                  <i className={`fas ${ICONS.warning}`}></i> Обучать даже при изъянах в датасете
                </span>
                <span className="skip-check-hint">Игнорировать вердикт аналитика данных, если он забраковал датасет.</span>
              </span>
              <span className="skip-check-switch">
                <input
                  type="checkbox"
                  checked={skipDatasetCheck}
                  onChange={(e) => setSkipDatasetCheck(e.target.checked)}
                  disabled={submitting}
                />
                <span className="skip-check-track"><span className="skip-check-thumb" /></span>
              </span>
            </label>
          )}
          <div className="form-field full-width">
            <label>Модель <span className="required-star">*</span></label>
            <div className="model-mode-selector" role="radiogroup" aria-label="Модель">
              {MODEL_MODES.map(m => (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={modelMode === m.id}
                  className={`model-mode-option${modelMode === m.id ? ' active' : ''}`}
                  onClick={() => setModelMode(m.id)}
                  disabled={submitting}
                >
                  <span className="workflow-option-label">{m.label}</span>
                  <span className="workflow-option-hint">{m.hint}</span>
                </button>
              ))}
            </div>
          </div>
          {modelMode === 'new' ? (
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
          ) : (
            <div className="form-field">
              <label htmlFor="run-existing-model">Существующая модель <span className="required-star">*</span></label>
              <Combobox
                id="run-existing-model"
                icon={`fas ${ICONS.model}`}
                placeholder="выберите модель из реестра"
                value={existingModelName}
                options={models.map(m => m.name)}
                onChange={setExistingModelName}
                disabled={submitting}
              />
              <span className="field-hint">
                {matchedModel
                  ? `Версий: ${matchedModel.versions.length} — агенты обучат следующую.`
                  : 'Новые версии будут созданы под выбранной моделью.'}
              </span>
            </div>
          )}
          {workflow === 'development' && (
            <div className="form-field">
              <label htmlFor="run-max-iter">Попыток обучения (0 = агент решает сам)</label>
              <input
                id="run-max-iter"
                className="no-spinner"
                type="number"
                min={0}
                value={maxIter}
                onChange={(e) => setMaxIter(Math.max(0, Number(e.target.value) || 0))}
                disabled={submitting}
              />
            </div>
          )}
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className={`fas ${ICONS.info}`}></i> Контекст для агентов</h3>
          <p className="form-section-hint">Эту информацию агенты используют при подборе решения. Поля опциональны: без них агенты минимизируют затраты и максимизируют качество сами.</p>
        </div>
        <div className="form-grid">
          <div className="form-field full-width">
            <label htmlFor="run-business">Бизнес-требования</label>
            <textarea
              id="run-business"
              autoComplete="off"
              placeholder="Что нужно бизнесу от модели. Если не указать — агенты сами максимизируют качество"
              value={businessRequirements}
              onChange={(e) => setBusinessRequirements(e.target.value)}
              rows={3}
              disabled={submitting}
            />
          </div>
          <div className="form-field full-width">
            <label htmlFor="run-deployment">Ограничения прода</label>
            <textarea
              id="run-deployment"
              autoComplete="off"
              placeholder="Технические ограничения, например: только CPU, ≤100 МБ. Если не указать — агенты сами минимизируют затраты"
              value={deploymentConstraints}
              onChange={(e) => setDeploymentConstraints(e.target.value)}
              rows={3}
              disabled={submitting}
            />
          </div>
          {workflow === 'development' && (
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
          )}
        </div>
      </div>

      <div className="form-section">
        <div className="form-section-head">
          <h3><i className={`fas ${ICONS.tag}`}></i> Параметры запуска</h3>
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
            <><i className={`fas ${ICONS.loading} fa-spin`}></i> Запуск...</>
          ) : (
            <><i className={`fas ${ICONS.play}`}></i> Запустить</>
          )}
        </button>
      </div>
    </form>
  );
};

export default RunPipelineForm;
