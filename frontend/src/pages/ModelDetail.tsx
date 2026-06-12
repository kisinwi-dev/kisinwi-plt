import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { mlModelsService } from '../services/mlModelsService';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import type { MLModelVersion, MLModelFile } from '../types/mlModels';
import type { Dataset } from '../types/dataset';
import { formatBytes, formatDateTime, formatDateParts } from '../utils/format';
import { useCopyToClipboard, useModelMetricsStream } from '../hooks';
import { CollapseChevron, getDisclosureProps } from '../components/common/Collapse';
import ConfirmModal from '../components/common/ConfirmModal';
import { Tooltip } from '../components/common/Tooltip';
import Select from '../components/common/Select';
import { ModelMetricsCharts } from '../components/models';
import { ICONS } from '../constants/icons';
import { modelStatusLabel, statusBadgeClass } from '../constants';
import './Models.css';

// Финальные статусы реестра: после них перечитывать версию уже незачем.
const FINAL_REGISTRY_STATUSES = ['completed', 'failed', 'cancelled'];

/**
 * Описание модели: длинный текст сворачивается до clamp; кнопка-переключатель
 * появляется, только если текст реально не помещается. Монтируется
 * с key={model.id} — при смене версии состояние сбрасывается ремоунтом.
 */
const ModelDescription: React.FC<{ text: string }> = ({ text }) => {
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const ref = useRef<HTMLParagraphElement>(null);

  // Переполнение проверяем через ResizeObserver: пересчитывается и при
  // изменении ширины окна, когда clamp начинает/перестаёт резать текст.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(() => {
      setOverflows(el.scrollHeight > el.clientHeight + 1);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section className="detail-section">
      <h3 className="detail-section-title"><i className={`fas ${ICONS.description}`}></i> Описание</h3>
      <p
        ref={ref}
        className={`model-detail-description${expanded ? '' : ' model-detail-description--clamped'}`}
      >
        {text}
      </p>
      {(overflows || expanded) && (
        <button
          type="button"
          className="model-detail-description-toggle"
          onClick={() => setExpanded(v => !v)}
        >
          <i className={`fas ${expanded ? ICONS.collapse : ICONS.expand}`}></i>
          {expanded ? 'Свернуть' : 'Показать полностью'}
        </button>
      )}
    </section>
  );
};

const ModelDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotification();

  const [model, setModel] = useState<MLModelVersion | null>(null);
  // Все версии родительской модели — для переключателя в шапке.
  const [versions, setVersions] = useState<MLModelVersion[]>([]);
  const [files, setFiles] = useState<MLModelFile[]>([]);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  // loading выводится из id: пока загруженный id отстаёт от текущего
  // (первый заход, переключение версии), показываем лоадер.
  const [loadedId, setLoadedId] = useState<string | null>(null);
  const loading = !!id && loadedId !== id;
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [paramsOpen, setParamsOpen] = useState(false);

  // SSE-стрим метрик живёт здесь: графикам нужны данные, а шапке — сигнал
  // о завершении обучения. Одно соединение на страницу.
  const metricsStream = useModelMetricsStream(id ?? '');
  // Подтверждение удаления версии модели (модалка).
  const [pendingDelete, setPendingDelete] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    Promise.all([
      mlModelsService.getVersion(id),
      mlModelsService.getVersionFiles(id).catch(() => ({ files: [] })),
    ])
      .then(([modelData, filesData]) => {
        if (cancelled) return;
        setModel(modelData);
        setFiles(filesData.files);
        datasetService.getDataset(modelData.dataset_id)
          .then((ds) => { if (!cancelled) setDataset(ds); })
          .catch(() => {});
        // Список версий не критичен для страницы: при ошибке остаёмся с бейджем.
        mlModelsService.getVersions({ model_id: modelData.model_id })
          .then((res) => { if (!cancelled) setVersions(res.versions); })
          .catch(() => {});
      })
      .catch((error) => {
        if (cancelled) return;
        showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модель', 'error');
      })
      .finally(() => {
        if (!cancelled) setLoadedId(id);
      });

    return () => { cancelled = true; };
  }, [id, showNotification]);

  // Статус «обучена / не обучена» — только из реестра ml_models: metrics видит
  // лишь ход обучения. Завершение стрима — сигнал перечитать версию из реестра,
  // чтобы бейдж обновился без перезагрузки страницы. Статус читаем через ref:
  // зависимость от model зациклила бы запросы, если реестр отстаёт от стрима.
  // Ref обновляется эффектом без deps, объявленным до эффекта перечитывания, —
  // к моменту его запуска статус всегда свежий.
  const modelStatusRef = useRef<string | null>(null);
  useEffect(() => {
    modelStatusRef.current = model?.status ?? null;
  });
  useEffect(() => {
    if (!id || !metricsStream.finished) return;
    const status = modelStatusRef.current;
    if (!status || FINAL_REGISTRY_STATUSES.includes(status)) return;
    let cancelled = false;
    mlModelsService.getVersion(id)
      .then((fresh) => { if (!cancelled) setModel(fresh); })
      .catch(() => { /* реестр недоступен — остаёмся со старым снимком */ });
    return () => { cancelled = true; };
  }, [id, metricsStream.finished]);

  const handleCopyId = useCopyToClipboard();

  const handleDeleteVersion = async () => {
    if (!model) return;
    setPendingDelete(false);
    try {
      setBusy(true);
      await mlModelsService.deleteVersion(model.id);
      showNotification('Версия модели удалена', 'success');
      navigate('/models');
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось удалить версию', 'error');
      setBusy(false);
    }
  };

  const handleDownload = async (file: MLModelFile) => {
    setDownloadingId(file.id);
    try {
      await mlModelsService.downloadFile(file.id, file.filename);
    } catch (error) {
      showNotification(error instanceof Error ? error.message : 'Не удалось скачать файл', 'error');
    } finally {
      setDownloadingId(null);
    }
  };

  if (loading) {
    return (
      <div className="page">
        <div className="loading-state">
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка модели…
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="page">
        <button className="detail-back-link" onClick={() => navigate('/models')}>
          <i className={`fas ${ICONS.back}`}></i> К списку моделей
        </button>
        <div className="empty-state">
          <i className={`fas ${ICONS.notFound}`}></i> Модель не найдена.
        </div>
      </div>
    );
  }

  const trainParams = Object.keys(model.train_params ?? {}).length > 0;
  const datasetVersionName = dataset?.versions.find(v => v.id === model.dataset_version_id)?.name;

  const versionOptions = [...versions]
    .sort((a, b) => b.version - a.version)
    .map((v) => ({
      value: v.id,
      label: `v${v.version} · ${formatDateParts(v.created_at).date} · ${modelStatusLabel(v.status)}`,
    }));

  return (
    <div className="page model-detail">
      <button className="detail-back-link" onClick={() => navigate('/models')}>
        <i className={`fas ${ICONS.back}`}></i> К списку моделей
      </button>

      <header className="model-detail-header">
        <div className="model-detail-heading">
          <div className="model-detail-title">
            <h1>{model.name}</h1>
            {versionOptions.length > 1 ? (
              <div className="model-detail-version-select">
                <Select
                  icon={`fas ${ICONS.version}`}
                  ariaLabel="Версия модели"
                  value={model.id}
                  options={versionOptions}
                  onChange={(versionId) => {
                    if (versionId !== model.id) navigate(`/models/${versionId}`);
                  }}
                />
              </div>
            ) : (
              <span className="model-version"><i className={`fas ${ICONS.version}`}></i> v{model.version}</span>
            )}
            <span className={statusBadgeClass(model.status)}>
              {(model.status === 'in_progress' || model.status === 'training') && (
                <><i className={`fas ${ICONS.loading} fa-spin`}></i>{' '}</>
              )}
              {modelStatusLabel(model.status)}
            </span>
          </div>
          <Tooltip content="Нажмите, чтобы скопировать ID">
            <span
              className="model-detail-id"
              onClick={() => handleCopyId(model.id)}
            >
              <i className={`fas ${ICONS.id}`}></i>{model.id}
              <i className={`fas ${ICONS.copy} model-detail-id-copy-icon`}></i>
            </span>
          </Tooltip>
        </div>
        <div className="model-detail-actions">
          <button
            className="button secondary small"
            onClick={() => navigate(`/models/compare?ids=${model.id}`)}
          >
            <i className={`fas ${ICONS.compare}`}></i> Сравнить
          </button>
          <button
            className="button danger model-detail-delete"
            onClick={() => setPendingDelete(true)}
            disabled={busy}
          >
            <i className={`fas ${ICONS.delete}`}></i> Удалить версию
          </button>
        </div>
      </header>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.info}`}></i> Общая информация</h3>
        <div className="detail-fields">
          <div className="detail-field">
            <span className="detail-label"><i className={`fas ${ICONS.tag}`}></i> Тип</span>
            <span>{model.model_type || '—'}</span>
          </div>
          <div className="detail-field">
            <span className="detail-label"><i className={`fas ${ICONS.dateCreated}`}></i> Создана</span>
            <span>{formatDateTime(model.created_at)}</span>
          </div>
          <div className="detail-field">
            <span className="detail-label"><i className={`fas ${ICONS.framework}`}></i> Framework</span>
            <span>{model.framework ?? '—'}{model.framework_version ? ` ${model.framework_version}` : ''}</span>
          </div>
          <div className="detail-field">
            <span className="detail-label"><i className={`fas ${ICONS.dataset}`}></i> Датасет</span>
            <Tooltip content={model.dataset_id}>
              <button
                className="detail-link"
                onClick={() => navigate(`/datasets/${model.dataset_id}`)}
              >
                <i className={`fas ${ICONS.external}`}></i>
                {dataset ? dataset.name : model.dataset_id}
              </button>
            </Tooltip>
          </div>
          <div className="detail-field">
            <span className="detail-label"><i className={`fas ${ICONS.version}`}></i> Версия датасета</span>
            <Tooltip content={model.dataset_version_id}>
              <button
                className="detail-link"
                onClick={() => navigate(
                  datasetVersionName
                    ? `/datasets/${model.dataset_id}?version=${encodeURIComponent(datasetVersionName)}`
                    : `/datasets/${model.dataset_id}`,
                )}
              >
                <i className={`fas ${ICONS.external}`}></i>
                {datasetVersionName ?? model.dataset_version_id}
              </button>
            </Tooltip>
          </div>
          <div className="detail-field detail-field--full">
            <span className="detail-label"><i className={`fas ${ICONS.classes}`}></i> Классы ({model.classes.length})</span>
            {model.classes.length > 0 ? (
              <div className="tag-list">
                {model.classes.map((cls) => (
                  <span key={cls} className="tag">{cls}</span>
                ))}
              </div>
            ) : (
              <span className="detail-empty">Классы не указаны.</span>
            )}
          </div>
        </div>
      </section>

      {model.description && <ModelDescription key={model.id} text={model.description} />}

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.metrics}`}></i> Метрики</h3>
        <ModelMetricsCharts modelId={model.id} metricsReport={model.metrics_report} stream={metricsStream} />
      </section>

      <section className="detail-section">
        <div
          className="detail-section-collapsible-header"
          {...getDisclosureProps(paramsOpen, () => setParamsOpen(o => !o))}
        >
          <CollapseChevron open={paramsOpen} />
          <h3 className="detail-section-title" style={{ margin: 0 }}>
            <i className={`fas ${ICONS.trainingParams}`}></i> Параметры обучения
          </h3>
        </div>
        {paramsOpen && (
          trainParams ? (
            <pre className="detail-pre detail-pre--params">
              {JSON.stringify(model.train_params, null, 2)}
            </pre>
          ) : (
            <p className="detail-empty detail-empty--params">Параметры обучения не заданы.</p>
          )
        )}
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.weights}`}></i> Файлы весов ({files.length})</h3>
        {files.length > 0 ? (
          <div className="model-files">
            {files.map((file) => (
              <div key={file.id} className="model-file-row">
                <div className="model-file-info">
                  <span className="model-file-name"><i className={`fas ${ICONS.file}`}></i> {file.filename}</span>
                  <span className="model-file-meta">
                    {formatBytes(file.file_size)} · {formatDateTime(file.created_at)}
                  </span>
                </div>
                <button
                  className="button"
                  disabled={downloadingId === file.id}
                  onClick={() => handleDownload(file)}
                >
                  {downloadingId === file.id ? (
                    <><i className={`fas ${ICONS.loading} fa-spin`}></i> Скачивание…</>
                  ) : (
                    <><i className={`fas ${ICONS.download}`}></i> Скачать</>
                  )}
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="detail-empty">У модели нет файлов весов.</p>
        )}
      </section>

      <ConfirmModal
        open={pendingDelete}
        danger
        title="Удалить версию модели?"
        message={`Версия v${model.version} модели «${model.name}» будет удалена безвозвратно вместе с файлами весов.`}
        confirmLabel="Удалить"
        cancelLabel="Отмена"
        onConfirm={handleDeleteVersion}
        onCancel={() => setPendingDelete(false)}
      />
    </div>
  );
};

export default ModelDetail;
