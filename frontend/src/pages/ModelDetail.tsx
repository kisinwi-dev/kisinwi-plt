import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { mlModelsService } from '../services/mlModelsService';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import type { MLModelVersion, MLModelFile } from '../types/mlModels';
import type { Dataset } from '../types/dataset';
import { formatBytes, formatDateTime } from '../utils/format';
import { useCopyToClipboard } from '../hooks';
import { CollapseChevron, getDisclosureProps } from '../components/common/Collapse';
import { Tooltip } from '../components/common/Tooltip';
import { ModelMetricsCharts } from '../components/models';
import { ICONS } from '../constants/icons';
import './Models.css';

const ModelDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotification();

  const [model, setModel] = useState<MLModelVersion | null>(null);
  const [files, setFiles] = useState<MLModelFile[]>([]);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [paramsOpen, setParamsOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setLoading(true);

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
      })
      .catch((error) => {
        if (cancelled) return;
        showNotification(error instanceof Error ? error.message : 'Не удалось загрузить модель', 'error');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [id, showNotification]);

  const handleCopyId = useCopyToClipboard();

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

  return (
    <div className="page model-detail">
      <button className="detail-back-link" onClick={() => navigate('/models')}>
        <i className={`fas ${ICONS.back}`}></i> К списку моделей
      </button>

      <div className="model-detail-header">
        <div className="model-detail-title">
          <h1>{model.name}</h1>
          <span className="model-version"><i className={`fas ${ICONS.version}`}></i> v{model.version}</span>
          <span className={`status-badge status-${model.status}`}>{model.status}</span>
          <button
            className="button secondary small"
            onClick={() => navigate(`/models/compare?from=${model.id}`)}
          >
            <i className={`fas ${ICONS.compare}`}></i> Сравнить
          </button>
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
        {model.description && <p className="model-detail-description">{model.description}</p>}
      </div>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.info}`}></i> Общая информация</h3>
        <div className="detail-fields">
          <div className="detail-field"><span className="detail-label">Тип</span><span>{model.model_type || '—'}</span></div>
          <div className="detail-field"><span className="detail-label">Framework</span><span>{model.framework ?? '—'}{model.framework_version ? ` ${model.framework_version}` : ''}</span></div>
          <div className="detail-field"><span className="detail-label">Создана</span><span>{formatDateTime(model.created_at)}</span></div>
          <div className="detail-field">
            <span className="detail-label">Датасет</span>
            <Tooltip content={model.dataset_id}>
              <button
                className="detail-link"
                onClick={() => navigate('/datasets')}
              >
                <i className={`fas ${ICONS.dataset}`}></i>
                {dataset ? dataset.name : model.dataset_id}
              </button>
            </Tooltip>
          </div>
          <div className="detail-field">
            <span className="detail-label">Версия датасета</span>
            <Tooltip content={model.dataset_version_id}>
              <button
                className="detail-link"
                onClick={() => navigate('/datasets')}
              >
                <i className={`fas ${ICONS.version}`}></i>
                {dataset?.versions.find(v => v.id === model.dataset_version_id)?.name ?? model.dataset_version_id}
              </button>
            </Tooltip>
          </div>
        </div>
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.classes}`}></i> Классы ({model.classes.length})</h3>
        {model.classes.length > 0 ? (
          <div className="tag-list">
            {model.classes.map((cls) => (
              <span key={cls} className="tag">{cls}</span>
            ))}
          </div>
        ) : (
          <p className="detail-empty">Классы не указаны.</p>
        )}
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.metrics}`}></i> Метрики</h3>
        <ModelMetricsCharts modelId={model.id} />
        {model.metrics_report && (
          <div className="metrics-report-details">
            <div
              className="metrics-report-summary"
              {...getDisclosureProps(reportOpen, () => setReportOpen(o => !o))}
            >
              <CollapseChevron open={reportOpen} />
              <i className={`fas ${ICONS.report}`}></i> Текстовый отчёт
            </div>
            {reportOpen && <pre className="detail-pre metrics-report-pre">{model.metrics_report}</pre>}
          </div>
        )}
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
    </div>
  );
};

export default ModelDetail;
