import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { mlModelsService } from '../services/mlModelsService';
import { useNotification } from '../contexts/NotificationContext';
import type { MLModel, MLModelFile } from '../types/mlModels';
import { formatBytes, formatDateTime } from '../utils/format';
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
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

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
      <div className="models-page">
        <div className="models-status-message">
          <i className="fas fa-spinner fa-spin"></i> Загрузка модели…
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="models-page">
        <button className="button secondary" onClick={() => navigate('/models')}>
          <i className="fas fa-arrow-left"></i> К списку
        </button>
        <div className="models-status-message">
          <i className="fas fa-triangle-exclamation"></i> Модель не найдена.
        </div>
      </div>
    );
  }

  const trainParams = Object.entries(model.train_params ?? {});

  return (
    <div className="models-page model-detail">
      <button className="button secondary back-button" onClick={() => navigate('/models')}>
        <i className="fas fa-arrow-left"></i> К списку
      </button>

      <div className="models-header">
        <div className="model-detail-title">
          <h1>{model.name}</h1>
          <span className="model-version"><i className="fas fa-code-branch"></i> v{model.version}</span>
          <span className={`status-badge status-${model.status}`}>{model.status}</span>
        </div>
        {model.description && <p className="models-description">{model.description}</p>}
      </div>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className="fas fa-circle-info"></i> Общая информация</h3>
        <div className="detail-fields">
          <div className="detail-field"><span className="detail-label">Тип</span><span>{model.model_type || '—'}</span></div>
          <div className="detail-field"><span className="detail-label">Framework</span><span>{model.framework ?? '—'}{model.framework_version ? ` ${model.framework_version}` : ''}</span></div>
          <div className="detail-field"><span className="detail-label">Создана</span><span>{formatDateTime(model.created_at)}</span></div>
          <div className="detail-field"><span className="detail-label">Dataset ID</span><span className="mono">{model.dataset_id}</span></div>
          <div className="detail-field"><span className="detail-label">Dataset version ID</span><span className="mono">{model.dataset_version_id}</span></div>
          <div className="detail-field"><span className="detail-label">Model ID</span><span className="mono">{model.id}</span></div>
        </div>
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className="fas fa-tags"></i> Классы ({model.classes.length})</h3>
        {model.classes.length > 0 ? (
          <div className="model-classes">
            {model.classes.map((cls) => (
              <span key={cls} className="class-tag">{cls}</span>
            ))}
          </div>
        ) : (
          <p className="detail-empty">Классы не указаны.</p>
        )}
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className="fas fa-chart-line"></i> Метрики</h3>
        {model.metrics_report ? (
          <pre className="detail-pre">{model.metrics_report}</pre>
        ) : (
          <p className="detail-empty">Отчёт по метрикам отсутствует.</p>
        )}
      </section>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className="fas fa-sliders"></i> Параметры обучения</h3>
        {trainParams.length > 0 ? (
          <div className="detail-fields">
            {trainParams.map(([key, value]) => (
              <div key={key} className="detail-field">
                <span className="detail-label">{key}</span>
                <span className="mono">{renderParamValue(value)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="detail-empty">Параметры обучения не заданы.</p>
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
