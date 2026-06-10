import React, { useEffect, useMemo, useState } from 'react';
import { datasetService } from '../../services/datasetService';
import { formatBytes, formatDateTime } from '../../utils/format';
import { ICONS } from '../../constants/icons';
import type { ImageSizeStats, SplitStats, Version, VersionSplitsResponse } from '../../types/dataset';
import './VersionStatsModal.css';

interface Props {
  datasetId: string;
  version: Version;
  isDefault: boolean;
  onClose: () => void;
}

const SPLIT_LABELS: Record<string, string> = {
  train: 'Train',
  val: 'Val',
  test: 'Test',
};

const SPLIT_ORDER = ['train', 'val', 'test'];

const BALANCE_THRESHOLD = 0.7;

// Псевдо-таб режима сравнения сплитов; не пересекается с именами реальных сплитов.
const COMPARE_TAB = '__compare__';

// Цвет сплита по его индексу в splitKeys; токены заданы в :root theme.css
// и не переопределяются темами.
const SPLIT_COLORS = ['var(--color-accent)', 'var(--color-info)', 'var(--color-warning)'];

// Пороги дрейфа распределения между сплитами, в процентных пунктах.
const DRIFT_WARN_PP = 5;
const DRIFT_BAD_PP = 10;

// Известные сплиты — в фиксированном порядке, неизвестные ключи — после них.
const orderSplitKeys = (keys: string[]): string[] =>
  [...keys].sort((a, b) => {
    const ia = SPLIT_ORDER.indexOf(a);
    const ib = SPLIT_ORDER.indexOf(b);
    return (ia === -1 ? SPLIT_ORDER.length : ia) - (ib === -1 ? SPLIT_ORDER.length : ib);
  });

const balanceTone = (ratio: number) => (ratio >= BALANCE_THRESHOLD ? 'good' : 'poor');

interface StatCardProps {
  icon: string;
  label: string;
  value: React.ReactNode;
  tone?: 'good' | 'poor';
  children?: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({ icon, label, value, tone, children }) => (
  <div className="vstats-card">
    <span className="vstats-card-label">
      <i className={`fas ${icon}`}></i> {label}
    </span>
    <span className={`vstats-card-value${tone ? ` vstats-balance--${tone}` : ''}`}>{value}</span>
    {children}
  </div>
);

interface BarRowProps {
  label: string;
  count: number;
  maxCount: number;
  pct?: string;
}

// Ширина бара масштабируется от максимального count в списке, чтобы самая
// крупная строка занимала всю дорожку; настоящий процент показываем текстом.
const BarRow: React.FC<BarRowProps> = ({ label, count, maxCount, pct }) => (
  <div className="vstats-bar-row">
    <span className="vstats-bar-label" title={label}>{label}</span>
    <span className="vstats-bar-track">
      <span
        className="vstats-bar-fill"
        style={{ width: `${maxCount > 0 ? (count / maxCount) * 100 : 0}%` }}
      />
    </span>
    <span className="vstats-bar-count">{count.toLocaleString()}</span>
    <span className="vstats-bar-pct">{pct ?? ''}</span>
  </div>
);

interface SplitPanelProps {
  stats: SplitStats;
  sizeInfo?: ImageSizeStats;
}

const SplitPanel: React.FC<SplitPanelProps> = ({ stats, sizeInfo }) => {
  const classes = stats.class_distribution;
  const maxClassCount = classes.length > 0 ? Math.max(...classes.map(cd => cd.count)) : 0;
  const sizeEntries = Object.entries(sizeInfo?.top_10_sizes ?? {}).sort((a, b) => b[1] - a[1]);
  const maxSizeCount = sizeEntries.length > 0 ? Math.max(...sizeEntries.map(([, n]) => n)) : 0;
  const tone = balanceTone(stats.balance_ratio);

  return (
    <div className="vstats-split-panel">
      <div className="vstats-chips">
        <span className="vstats-chip">
          <i className={`fas ${ICONS.samples}`}></i> {stats.total_samples.toLocaleString()} сэмплов
        </span>
        <span className="vstats-chip">
          <i className={`fas ${ICONS.classes}`}></i> {stats.num_classes} классов
        </span>
        <span className="vstats-chip">
          <i className={`fas ${ICONS.balance}`}></i> баланс {(stats.balance_ratio * 100).toFixed(0)}%
        </span>
        <span className={`vstats-chip vstats-chip--${tone}`}>
          <i className={`fas ${stats.is_balanced ? ICONS.success : ICONS.error}`}></i>
          {stats.is_balanced ? 'сбалансирован' : 'несбалансирован'}
        </span>
      </div>

      <h5 className="vstats-subtitle">
        <i className={`fas ${ICONS.classes}`}></i> Распределение классов
      </h5>
      {classes.length === 0 ? (
        <p className="vstats-empty">Нет данных о классах.</p>
      ) : (
        <div className="vstats-bar-list">
          <div className="vstats-bar-row vstats-bar-head">
            <span className="vstats-bar-label">Класс</span>
            <span className="vstats-bar-track-head"></span>
            <span className="vstats-bar-count">Кол-во</span>
            <span className="vstats-bar-pct">%</span>
          </div>
          {classes.map(cd => (
            <BarRow
              key={cd.class_name}
              label={cd.class_name}
              count={cd.count}
              maxCount={maxClassCount}
              pct={cd.percentage != null ? `${cd.percentage.toFixed(1)}%` : '—'}
            />
          ))}
        </div>
      )}

      {sizeInfo && sizeInfo.total_images > 0 && (
        <>
          <h5 className="vstats-subtitle">
            <i className={`fas ${ICONS.imageSize}`}></i> Размеры изображений
          </h5>
          <div className="vstats-chips">
            {sizeInfo.most_common_size && (
              <span className="vstats-chip">частый размер: {sizeInfo.most_common_size}</span>
            )}
            <span className="vstats-chip">{sizeInfo.unique_sizes} уникальных</span>
            {sizeInfo.size_consistency != null && (
              <span className={`vstats-chip vstats-chip--${sizeInfo.size_consistency >= 0.9 ? 'good' : 'neutral'}`}>
                единообразие {(sizeInfo.size_consistency * 100).toFixed(0)}%
              </span>
            )}
          </div>
          {sizeEntries.length > 0 && (
            <div className="vstats-bar-list vstats-bar-list--compact">
              {sizeEntries.map(([size, count]) => (
                <BarRow
                  key={size}
                  label={size}
                  count={count}
                  maxCount={maxSizeCount}
                  pct={`${((count / sizeInfo.total_images) * 100).toFixed(1)}%`}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

interface SplitComparePanelProps {
  splitsSummary: Record<string, SplitStats>;
  splitKeys: string[];
}

// Сравнение распределений классов между сплитами: проценты внутри сплита
// (размеры сплитов разные, абсолютные count сравнивать нечестно) и
// Δ — разброс процента класса между сплитами в процентных пунктах.
const SplitComparePanel: React.FC<SplitComparePanelProps> = ({ splitsSummary, splitKeys }) => {
  // Union классов: порядок первого сплита, недостающие из остальных — в конец.
  const classNames: string[] = [];
  const seen = new Set<string>();
  for (const key of splitKeys) {
    for (const cd of splitsSummary[key].class_distribution) {
      if (!seen.has(cd.class_name)) {
        seen.add(cd.class_name);
        classNames.push(cd.class_name);
      }
    }
  }

  const pctBySplit = splitKeys.map(key =>
    new Map(splitsSummary[key].class_distribution.map(cd => [cd.class_name, cd.percentage ?? 0])),
  );

  const rows = classNames.map(name => {
    const pcts = pctBySplit.map(m => m.get(name) ?? 0);
    return { name, pcts, delta: Math.max(...pcts) - Math.min(...pcts) };
  });

  if (rows.length === 0) {
    return <p className="vstats-empty">Нет данных о классах.</p>;
  }

  // Общий масштаб баров для всей таблицы, чтобы сравнение было честным.
  const maxPct = Math.max(...rows.flatMap(r => r.pcts), 0);

  const deltaClass = (delta: number) =>
    delta > DRIFT_BAD_PP ? ' vstats-compare-delta--bad'
    : delta > DRIFT_WARN_PP ? ' vstats-compare-delta--warn'
    : '';

  return (
    <div className="vstats-split-panel">
      <div className="vstats-compare-legend">
        {splitKeys.map((key, i) => (
          <span key={key} className="vstats-legend-item">
            <span
              className="vstats-legend-dot"
              style={{ background: SPLIT_COLORS[i % SPLIT_COLORS.length] }}
            />
            {SPLIT_LABELS[key] ?? key}
            <span className="vstats-legend-count">
              {splitsSummary[key].total_samples.toLocaleString()}
            </span>
          </span>
        ))}
      </div>
      <div className="vstats-bar-list">
        <div className="vstats-compare-row vstats-bar-head">
          <span className="vstats-bar-label">Класс</span>
          <span className="vstats-bar-label">Доля в сплите</span>
          <span
            className="vstats-compare-delta"
            title={`Разброс доли класса между сплитами в процентных пунктах (макс. % − мин. %). Больше ${DRIFT_WARN_PP} п.п. — заметный дрейф, больше ${DRIFT_BAD_PP} п.п. — сильный.`}
          >
            Δ п.п. <i className={`fas ${ICONS.info}`}></i>
          </span>
        </div>
        {rows.map(row => (
          <div key={row.name} className="vstats-compare-row">
            <span className="vstats-bar-label" title={row.name}>{row.name}</span>
            <span className="vstats-compare-bars">
              {row.pcts.map((pct, i) => (
                <span key={splitKeys[i]} className="vstats-compare-line">
                  <span className="vstats-bar-track">
                    <span
                      className="vstats-bar-fill"
                      style={{
                        width: `${maxPct > 0 ? (pct / maxPct) * 100 : 0}%`,
                        background: SPLIT_COLORS[i % SPLIT_COLORS.length],
                      }}
                    />
                  </span>
                  <span className="vstats-compare-pct">{pct.toFixed(1)}%</span>
                </span>
              ))}
            </span>
            <span className={`vstats-compare-delta${deltaClass(row.delta)}`}>
              {row.delta.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const VersionStatsModal: React.FC<Props> = ({ datasetId, version, isDefault, onClose }) => {
  const [splits, setSplits] = useState<VersionSplitsResponse | null>(null);
  const [status, setStatus] = useState<'loading' | 'error' | 'ready'>('loading');
  const [retryKey, setRetryKey] = useState(0);
  const [selectedSplit, setSelectedSplit] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    datasetService.getVersionSplits(datasetId, version.id)
      .then(data => {
        if (cancelled) return;
        setSplits(data);
        setStatus('ready');
      })
      .catch(() => {
        if (!cancelled) setStatus('error');
      });
    return () => { cancelled = true; };
  }, [datasetId, version.id, retryKey]);

  // Закрытие по Escape и блокировка скролла страницы, пока модалка открыта.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  const splitKeys = useMemo(
    () => (splits ? orderSplitKeys(Object.keys(splits.splits_summary)) : []),
    [splits],
  );

  // Активный таб — производное значение: выбранный пользователем сплит
  // (или режим сравнения), если он доступен, иначе первый сплит по порядку.
  const compareAvailable = splitKeys.length > 1;
  const activeSplit = selectedSplit
    && (splitKeys.includes(selectedSplit) || (selectedSplit === COMPARE_TAB && compareAvailable))
    ? selectedSplit
    : splitKeys[0] ?? null;

  const ready = status === 'ready' && splits !== null;
  const numClasses = ready && splitKeys.length > 0
    ? Math.max(...splitKeys.map(key => splits.splits_summary[key].num_classes))
    : null;
  const overallBalance = ready ? splits.overall_balance : null;

  const formatEntries = Object.entries(version.image_format_stats ?? {}).sort((a, b) => b[1] - a[1]);
  const totalFormatCount = formatEntries.reduce((sum, [, n]) => sum + n, 0);
  const maxFormatCount = formatEntries.length > 0 ? Math.max(...formatEntries.map(([, n]) => n)) : 0;
  const hasAboutSection = formatEntries.length > 0 || !!version.description || version.sources.length > 0;

  const currentSplit = ready && activeSplit && activeSplit !== COMPARE_TAB
    ? splits.splits_summary[activeSplit]
    : null;

  return (
    <div className="vstats-overlay" onClick={onClose}>
      <div
        className="vstats-modal"
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="vstats-header">
          <div className="vstats-header-icon">
            <i className={`fas ${ICONS.datasetStats}`}></i>
          </div>
          <div className="vstats-header-text">
            <h3>
              Статистика версии «{version.name}»
              {isDefault && <span className="vstats-default-badge">по умолчанию</span>}
            </h3>
            <span className="vstats-header-sub">
              <i className={`fas ${ICONS.dateCreated}`}></i> {formatDateTime(version.created_at)}
            </span>
          </div>
          <button className="icon-button vstats-close" onClick={onClose} title="Закрыть">
            <i className={`fas ${ICONS.close}`}></i>
          </button>
        </header>

        <div className="vstats-body">
          <div className="vstats-cards">
            <StatCard icon={ICONS.samples} label="Образцов" value={version.num_samples.toLocaleString()} />
            <StatCard icon={ICONS.dataset} label="Размер" value={formatBytes(version.size_bytes)} />
            <StatCard icon={ICONS.classes} label="Классов" value={numClasses ?? '—'} />
            <StatCard
              icon={ICONS.balance}
              label="Баланс классов"
              value={overallBalance != null ? `${(overallBalance * 100).toFixed(0)}%` : '—'}
              tone={overallBalance != null ? balanceTone(overallBalance) : undefined}
            >
              {overallBalance != null && (
                <span className="vstats-card-bar">
                  <span
                    className={`vstats-card-bar-fill vstats-card-bar-fill--${balanceTone(overallBalance)}`}
                    style={{ width: `${overallBalance * 100}%` }}
                  />
                </span>
              )}
            </StatCard>
          </div>

          <section className="vstats-section">
            <h4 className="vstats-section-title">
              <i className={`fas ${ICONS.version}`}></i> Сплиты
            </h4>
            {status === 'loading' && (
              <div className="vstats-placeholder">
                <i className={`fas ${ICONS.loading} fa-spin`}></i> Загрузка статистики…
              </div>
            )}
            {status === 'error' && (
              <div className="vstats-placeholder vstats-placeholder--error">
                <i className={`fas ${ICONS.warning}`}></i> Не удалось загрузить статистику сплитов.
                <button
                  className="button secondary small"
                  onClick={() => { setStatus('loading'); setRetryKey(k => k + 1); }}
                >
                  Повторить
                </button>
              </div>
            )}
            {ready && splitKeys.length === 0 && (
              <div className="vstats-placeholder">
                <i className={`fas ${ICONS.empty}`}></i> Статистика по сплитам недоступна.
              </div>
            )}
            {ready && splitKeys.length > 0 && (
              <>
                {compareAvailable && (
                  <div className="vstats-tabs">
                    {splitKeys.map(key => (
                      <button
                        key={key}
                        className={`vstats-tab${key === activeSplit ? ' vstats-tab--active' : ''}`}
                        onClick={() => setSelectedSplit(key)}
                      >
                        {SPLIT_LABELS[key] ?? key}
                        <span className="vstats-tab-count">
                          {splits.splits_summary[key].total_samples.toLocaleString()}
                        </span>
                      </button>
                    ))}
                    <button
                      className={`vstats-tab${activeSplit === COMPARE_TAB ? ' vstats-tab--active' : ''}`}
                      onClick={() => setSelectedSplit(COMPARE_TAB)}
                    >
                      <i className={`fas ${ICONS.balance}`}></i> Сравнение
                    </button>
                  </div>
                )}
                {currentSplit && activeSplit && (
                  <SplitPanel stats={currentSplit} sizeInfo={splits.image_size_stats?.[activeSplit]} />
                )}
                {activeSplit === COMPARE_TAB && (
                  <SplitComparePanel splitsSummary={splits.splits_summary} splitKeys={splitKeys} />
                )}
              </>
            )}
          </section>

          {hasAboutSection && (
            <section className="vstats-section">
              <h4 className="vstats-section-title">
                <i className={`fas ${ICONS.info}`}></i> О версии
              </h4>
              {formatEntries.length > 0 && (
                <>
                  <h5 className="vstats-subtitle">
                    <i className={`fas ${ICONS.fileImage}`}></i> Форматы изображений
                  </h5>
                  <div className="vstats-bar-list vstats-bar-list--compact">
                    {formatEntries.map(([format, count]) => (
                      <BarRow
                        key={format}
                        label={format}
                        count={count}
                        maxCount={maxFormatCount}
                        pct={totalFormatCount > 0 ? `${((count / totalFormatCount) * 100).toFixed(1)}%` : '—'}
                      />
                    ))}
                  </div>
                </>
              )}
              {version.description && (
                <>
                  <h5 className="vstats-subtitle">
                    <i className={`fas ${ICONS.description}`}></i> Описание
                  </h5>
                  <p className="vstats-description">{version.description}</p>
                </>
              )}
              {version.sources.length > 0 && (
                <>
                  <h5 className="vstats-subtitle">
                    <i className={`fas ${ICONS.link}`}></i> Источники ({version.sources.length})
                  </h5>
                  <div className="vstats-sources">
                    {version.sources.map((src, idx) => (
                      <div key={idx} className="vstats-source-item">
                        <span className="vstats-source-type">{src.type}</span>
                        {src.url ? (
                          <a
                            href={src.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="vstats-source-link"
                            title={src.url}
                          >
                            <i className={`fas ${ICONS.external}`}></i>
                            <span>{src.description || src.url}</span>
                          </a>
                        ) : (
                          <span className="vstats-source-text">{src.description || '—'}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </section>
          )}
        </div>
      </div>
    </div>
  );
};

export default VersionStatsModal;
