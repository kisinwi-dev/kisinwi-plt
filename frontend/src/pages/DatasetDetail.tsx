import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { datasetService } from '../services/datasetService';
import { useNotification } from '../contexts/NotificationContext';
import { useCopyToClipboard } from '../hooks';
import { formatBytes, formatDateTime } from '../utils/format';
import type { Dataset, SourceItem, VersionSplitsResponse } from '../types/dataset';
import { AddVersionForm, VersionSplitsStats } from '../components/datasets';
import ConfirmModal from '../components/common/ConfirmModal';
import { ICONS } from '../constants/icons';
import './Datasets.css';

const makeUploadId = () => `upload_${Date.now()}`;

const EMPTY_VERSION = () => ({
  name: '',
  description: '',
  sources: [] as SourceItem[],
  file: null as File | null,
});

const DatasetDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { showNotification } = useNotification();
  const copyToClipboard = useCopyToClipboard();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [showVersionForm, setShowVersionForm] = useState(false);
  const [versionFilter, setVersionFilter] = useState('');
  const [newVersion, setNewVersion] = useState(EMPTY_VERSION);
  const [versionStats, setVersionStats] = useState<Record<string, VersionSplitsResponse | 'loading' | 'error'>>({});

  // Ожидающее подтверждения удаление: весь датасет либо конкретная версия.
  const [pendingDelete, setPendingDelete] = useState<
    { kind: 'dataset' } | { kind: 'version'; id: string; name: string } | null
  >(null);

  const loadDataset = React.useCallback(async () => {
    if (!id) return;
    try {
      const data = await datasetService.getDataset(id);
      setDataset(data);
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Не удалось загрузить датасет', 'error');
    } finally {
      setLoading(false);
    }
  }, [id, showNotification]);

  useEffect(() => {
    const init = async () => {
      setLoading(true);
      await loadDataset();
    };
    init();
  }, [loadDataset]);

  const handleShowVersionStats = async (versionId: string) => {
    if (!id) return;
    if (versionStats[versionId]) {
      setVersionStats(prev => { const next = { ...prev }; delete next[versionId]; return next; });
      return;
    }
    setVersionStats(prev => ({ ...prev, [versionId]: 'loading' }));
    try {
      const data = await datasetService.getVersionSplits(id, versionId);
      setVersionStats(prev => ({ ...prev, [versionId]: data }));
    } catch {
      setVersionStats(prev => ({ ...prev, [versionId]: 'error' }));
    }
  };

  const handleAddVersion = async () => {
    if (!id) return;
    if (!newVersion.name) {
      showNotification('Введите название версии', 'warning');
      return;
    }
    try {
      setBusy(true);
      const id_data = makeUploadId();
      if (newVersion.file) {
        const uploaded = await datasetService.uploadFile(id_data, newVersion.file);
        if (!uploaded) throw new Error('Не удалось загрузить файл');
      }
      await datasetService.createVersion(id, {
        id_data,
        name: newVersion.name,
        description: newVersion.description,
        sources: newVersion.sources,
      });
      await loadDataset();
      setShowVersionForm(false);
      setNewVersion(EMPTY_VERSION);
      showNotification('Версия успешно добавлена', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка добавления версии', 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteVersion = async (versionId: string) => {
    if (!id) return;
    try {
      setBusy(true);
      const deleted = await datasetService.deleteVersion(id, versionId);
      if (!deleted) throw new Error('Не удалось удалить версию');
      await loadDataset();
      showNotification('Версия удалена', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления версии', 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleSetDefaultVersion = async (versionId: string) => {
    if (!id) return;
    try {
      setBusy(true);
      const ok = await datasetService.setDefaultVersion(id, versionId);
      if (!ok) throw new Error('Не удалось установить версию по умолчанию');
      await loadDataset();
      showNotification('Версия по умолчанию обновлена', 'success');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка установки версии', 'error');
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteDataset = async () => {
    if (!id) return;
    try {
      setBusy(true);
      const deleted = await datasetService.deleteDataset(id);
      if (!deleted) throw new Error('Не удалось удалить датасет');
      showNotification('Датасет удалён', 'success');
      navigate('/datasets');
    } catch (err) {
      showNotification(err instanceof Error ? err.message : 'Ошибка удаления датасета', 'error');
      setBusy(false);
    }
  };

  const handleConfirmDelete = () => {
    if (!pendingDelete) return;
    const target = pendingDelete;
    setPendingDelete(null);
    if (target.kind === 'dataset') {
      handleDeleteDataset();
    } else {
      handleDeleteVersion(target.id);
    }
  };

  const filteredVersions = useMemo(() => {
    const q = versionFilter.trim().toLowerCase();
    const versions = dataset?.versions ?? [];
    if (!q) return versions;
    return versions.filter(ver =>
      ver.name.toLowerCase().includes(q) || ver.description.toLowerCase().includes(q)
    );
  }, [dataset?.versions, versionFilter]);

  if (loading) {
    return (
      <div className="page">
        <div className="loading-state">
          <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка датасета…
        </div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="page">
        <button className="button secondary" onClick={() => navigate('/datasets')}>
          <i className={`fas ${ICONS.back}`}></i> К списку
        </button>
        <div className="empty-state">
          <i className={`fas ${ICONS.notFound}`}></i> Датасет не найден.
        </div>
      </div>
    );
  }

  return (
    <div className="page dataset-detail">
      <button className="detail-back-link" onClick={() => navigate('/datasets')}>
        <i className={`fas ${ICONS.back}`}></i> К списку датасетов
      </button>

      <header className="dataset-detail-header">
        <div className="dataset-detail-heading">
          <div className="dataset-detail-title">
            <h1>{dataset.name}</h1>
            <span className="dataset-badge">
              <i className={`fas ${ICONS.tag}`}></i> {dataset.type} / {dataset.task}
            </span>
          </div>
          <span
            className="dataset-id"
            title="Нажмите, чтобы скопировать ID"
            onClick={() => copyToClipboard(dataset.id)}
          >
            <i className={`fas ${ICONS.id}`}></i>{dataset.id}
            <i className={`fas ${ICONS.copy} dataset-id-copy-icon`}></i>
          </span>
        </div>
        <button
          className="button danger dataset-detail-delete"
          onClick={() => setPendingDelete({ kind: 'dataset' })}
          disabled={busy}
        >
          <i className={`fas ${ICONS.delete}`}></i> Удалить датасет
        </button>
      </header>

      <section className="detail-section">
        <h3 className="detail-section-title"><i className={`fas ${ICONS.info}`}></i> Общая информация</h3>
        <div className="detail-fields">
          <div className="detail-field"><span className="detail-label"><i className={`fas ${ICONS.datasetType}`}></i> Тип</span><span>{dataset.type || '—'}</span></div>
          <div className="detail-field"><span className="detail-label"><i className={`fas ${ICONS.taskTarget}`}></i> Задача</span><span>{dataset.task || '—'}</span></div>
          <div className="detail-field"><span className="detail-label"><i className={`fas ${ICONS.classes}`}></i> Классов</span><span>{dataset.classes_count}</span></div>
          <div className="detail-field"><span className="detail-label"><i className={`fas ${ICONS.version}`}></i> Версий</span><span>{dataset.versions.length}</span></div>
          <div className="detail-field"><span className="detail-label"><i className={`fas ${ICONS.dateCreated}`}></i> Создан</span><span>{formatDateTime(dataset.created_at)}</span></div>
          <div className="detail-field"><span className="detail-label"><i className={`fas ${ICONS.dateUpdated}`}></i> Обновлён</span><span>{formatDateTime(dataset.updated_at)}</span></div>
          <div className="detail-field detail-field--full">
            <span className="detail-label"><i className={`fas ${ICONS.classes}`}></i> Классы</span>
            {dataset.classes_names.length > 0 ? (
              <div className="tag-list">
                {dataset.classes_names.map(className => (
                  <span key={className} className="tag">{className}</span>
                ))}
              </div>
            ) : (
              <span className="detail-empty">Классы не указаны.</span>
            )}
          </div>
        </div>
      </section>

      {dataset.description && (
        <section className="detail-section">
          <h3 className="detail-section-title"><i className={`fas ${ICONS.description}`}></i> Описание</h3>
          <p className="dataset-detail-description">{dataset.description}</p>
        </section>
      )}

      <section className="detail-section versions-section">
        <div className="versions-section-head">
          <h3 className="detail-section-title"><i className={`fas ${ICONS.version}`}></i> Версии ({dataset.versions.length})</h3>
          <button
            className="button small"
            onClick={() => setShowVersionForm(v => !v)}
            disabled={busy}
          >
            <i className={`fas ${ICONS.add}`}></i> Добавить версию
          </button>
        </div>

        {showVersionForm && (
          <AddVersionForm
            version={newVersion}
            loading={busy}
            onVersionChange={setNewVersion}
            onSubmit={handleAddVersion}
            onCancel={() => setShowVersionForm(false)}
          />
        )}

        {dataset.versions.length > 1 && (
          <div className="filter-field versions-filter">
            <i className={`fas ${ICONS.search}`}></i>
            <input
              type="text"
              placeholder="Поиск по названию или описанию версии"
              value={versionFilter}
              onChange={(e) => setVersionFilter(e.target.value)}
            />
            {versionFilter && (
              <button
                type="button"
                className="versions-filter-clear"
                onClick={() => setVersionFilter('')}
                title="Сбросить фильтр"
              >
                <i className={`fas ${ICONS.close}`}></i>
              </button>
            )}
          </div>
        )}

        {dataset.versions.length === 0 ? (
          <p className="no-versions">Нет версий</p>
        ) : filteredVersions.length === 0 ? (
          <p className="no-versions">Версии не найдены</p>
        ) : (
          <div className="versions-list">
            {filteredVersions.map(ver => (
              <div
                key={ver.id}
                className={`version-item${ver.id === dataset.default_version_id ? ' version-item--default' : ''}`}
              >
                <div className="version-header">
                  <span className="version-name">
                    {ver.name}
                    {ver.id === dataset.default_version_id && (
                      <span className="default-badge-inline">по умолчанию</span>
                    )}
                  </span>
                  <div className="version-actions">
                    {ver.id !== dataset.default_version_id && (
                      <button
                        className="icon-button small"
                        onClick={() => handleSetDefaultVersion(ver.id)}
                        title="Сделать версией по умолчанию"
                        disabled={busy}
                      >
                        <i className={`fas ${ICONS.star}`}></i>
                      </button>
                    )}
                    <button
                      className="icon-button small"
                      onClick={() => handleShowVersionStats(ver.id)}
                      title="Статистика"
                    >
                      <i className={`fas ${ICONS.datasetStats}`}></i>
                    </button>
                    <button
                      className="icon-button small icon-button--danger"
                      onClick={() => setPendingDelete({ kind: 'version', id: ver.id, name: ver.name })}
                      title="Удалить версию"
                      disabled={busy}
                    >
                      <i className={`fas ${ICONS.delete}`}></i>
                    </button>
                  </div>
                </div>
                <div className="version-meta">
                  <span className="version-meta-item">
                    <span className="version-meta-label"><i className={`fas ${ICONS.dataset}`}></i> Размер</span>
                    <span className="version-meta-value">{formatBytes(ver.size_bytes)}</span>
                  </span>
                  <span className="version-meta-item">
                    <span className="version-meta-label"><i className={`fas ${ICONS.samples}`}></i> Образцов</span>
                    <span className="version-meta-value">{ver.num_samples.toLocaleString()}</span>
                  </span>
                  <span className="version-meta-item">
                    <span className="version-meta-label"><i className={`fas ${ICONS.dateCreated}`}></i> Загружено</span>
                    <span className="version-meta-value">{formatDateTime(ver.created_at)}</span>
                  </span>
                </div>
                {ver.description && (
                  <p className="version-description">
                    <span className="version-field-label">Описание:</span> {ver.description}
                  </p>
                )}
                {ver.sources.length > 0 && (
                  <div className="dataset-sources">
                    <span className="dataset-sources-label">
                      <i className={`fas ${ICONS.link}`}></i> Источники ({ver.sources.length})
                    </span>
                    {ver.sources.map((src, idx) => (
                      <div key={idx} className="source-item">
                        <span className="source-type-badge">{src.type}</span>
                        {src.url ? (
                          <a
                            href={src.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-link"
                            title={src.url}
                          >
                            <i className={`fas ${ICONS.external}`}></i>
                            <span className="source-link-text">{src.description || src.url}</span>
                          </a>
                        ) : (
                          <span className="source-description">{src.description || '—'}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {versionStats[ver.id] === 'loading' && (
                  <p className="stats-loading">Загрузка статистики...</p>
                )}
                {versionStats[ver.id] === 'error' && (
                  <p className="stats-error">Не удалось загрузить статистику</p>
                )}
                {typeof versionStats[ver.id] === 'object' && (
                  <VersionSplitsStats
                    stats={versionStats[ver.id] as VersionSplitsResponse}
                    onClose={() => handleShowVersionStats(ver.id)}
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </section>

      <ConfirmModal
        open={pendingDelete !== null}
        danger
        title={pendingDelete?.kind === 'version' ? 'Удалить версию?' : 'Удалить датасет?'}
        message={
          pendingDelete?.kind === 'version'
            ? `Версия «${pendingDelete.name}» будет удалена безвозвратно.`
            : pendingDelete?.kind === 'dataset'
            ? `Датасет «${dataset.name}» будет удалён безвозвратно вместе со всеми версиями.`
            : undefined
        }
        confirmLabel="Удалить"
        cancelLabel="Отмена"
        onConfirm={handleConfirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </div>
  );
};

export default DatasetDetail;
