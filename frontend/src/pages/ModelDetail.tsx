import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { mlModelsService } from '../services/mlModelsService';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import type { MLModel, MLModelFile } from '../types/mlModels';
import type { Dataset } from '../types/dataset';
import { formatBytes, formatDateTime } from '../utils/format';
import { ModelMetricsCharts } from '../components/models';
import './Models.css';

// Рендер значения train_params: примитивы — как есть, объекты/массивы — как JSON.
const renderParamValue = (value: unknown): string => {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
};

const ModelDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotification();

  const [model, setModel] = useState<MLModel | null>(null);
  const [files, setFiles] = useState<MLModelFile[]>([]);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [paramsOpen, setParamsOpen] = useState(false);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setLoading(true);

    Promise.all([
      mlModelsService.getModel(id),
      mlModelsService.getModelFiles(id).catch(() => ({ files: [] })),
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

  const handleCopyId = (value: string) => {
    navigator.clipboard.writeText(value);
    showNotification('ID скопирован', 'success');
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
          <i className="fas fa-spinner fa-spin"></i> Загрузка модели…
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="page">
        <button className="button secondary" onClick={() => navigate('/models')}>
          <i className="fas fa-arrow-left"></i> К списку
        </button>
        <div className="empty-state">
          <i className="fas fa-triangle-exclamation"></i> Модель не найдена.
        </div>
      </div>
    );
  }

  const trainParams = Object.keys(model.train_params ?? {}).length > 0;

  return (
    <div className="page model-detail">
      <button className="button secondary back-button" onClick={() => navigate('/models')}>
        <i className="fas fa-arrow-left"></i> К списку
      </button>

      <div className="model-detail-header">
        <div className="model-detail-title">
          <h1>{model.name}</h1>
          <span className="model-version"><i className="fas fa-code-branch"></i> v{model.version}</span>
          <span className={`status-badge status-${model.status}`}>{model.status}</span>
        </div>
        <span
          className="model-detail-id"
          title="Нажмите, чтобы скопировать ID"
          onClick={() => handleCopyId(model.id)}
        >
          <i className="fas fa-hashtag"></i>{model.id}
          <i className="fas fa-copy model-detail-id-copy-icon"></i>
        </span>
        {model.description && <p className="model-detail-description">{model.description}</p>}
      </div>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className="fas fa-circle-info"></i> Общая информация</h3>
        <div className="detail-fields">
          <div className="detail-field"><span className="detail-label">Тип</span><span>{model.model_type || '—'}</span></div>
          <div className="detail-field"><span className="detail-label">Framework</span><span>{model.framework ?? '—'}{model.framework_version ? ` ${model.framework_version}` : ''}</span></div>
          <div className="detail-field"><span className="detail-label">Создана</span><span>{formatDateTime(model.created_at)}</span></div>
          <div className="detail-field">
            <span className="detail-label">Датасет</span>
            <button
              className="detail-link"
              onClick={() => navigate('/datasets')}
              title={model.dataset_id}
            >
              <i className="fas fa-database"></i>
              {dataset ? dataset.name : model.dataset_id}
            </button>
          </div>
          <div className="detail-field">
            <span className="detail-label">Версия датасета</span>
            <button
              className="detail-link"
              onClick={() => navigate('/datasets')}
              title={model.dataset_version_id}
            >
              <i className="fas fa-code-branch"></i>
              {dataset?.versions.find(v => v.id === model.dataset_version_id)?.name ?? model.dataset_version_id}
            </button>
          </div>
        </div>
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className="fas fa-tags"></i> Классы ({model.classes.length})</h3>
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
        <h3 className="detail-section-title"><i className="fas fa-chart-line"></i> Метрики</h3>
        <ModelMetricsCharts modelId={model.id} />
        {model.metrics_report && (
          <details className="metrics-report-details">
            <summary className="metrics-report-summary">
              <i className="fas fa-file-lines"></i> Текстовый отчёт
            </summary>
            <pre className="detail-pre metrics-report-pre">{model.metrics_report}</pre>
          </details>
        )}
      </section>

      <section className="detail-section">
        <div className="detail-section-collapsible-header">
          <h3 className="detail-section-title" style={{ margin: 0 }}>
            <i className="fas fa-sliders"></i> Параметры обучения
          </h3>
          <button
            className={`detail-expand-btn${paramsOpen ? ' open' : ''}`}
            onClick={() => setParamsOpen(o => !o)}
            aria-expanded={paramsOpen}
          >
            <i className="fas fa-code"></i>
            {paramsOpen ? 'Скрыть JSON' : 'Показать JSON'}
            <i className={`fas fa-chevron-down detail-toggle-chevron${paramsOpen ? ' open' : ''}`}></i>
          </button>
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
        <h3 className="detail-section-title"><i className="fas fa-file-arrow-down"></i> Файлы весов ({files.length})</h3>
        {files.length > 0 ? (
          <div className="model-files">
            {files.map((file) => (
              <div key={file.id} className="model-file-row">
                <div className="model-file-info">
                  <span className="model-file-name"><i className="fas fa-file"></i> {file.filename}</span>
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
                    <><i className="fas fa-spinner fa-spin"></i> Скачивание…</>
                  ) : (
                    <><i className="fas fa-download"></i> Скачать</>
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
