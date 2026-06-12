import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import type { MLModelVersion } from '../../types/mlModels';
import type { ModelMetrics } from '../../services/metricsService';
import { datasetService } from '../../services/datasetService';
import type { Dataset } from '../../types/dataset';
import { ICONS } from '../../constants/icons';
import { Tooltip, InfoHint } from '../common/Tooltip';
import { formatDateParts } from '../../utils/format';
import type { CompareSide } from './modelCompare';
import { totalEpochsOf } from './modelCompare';
import CompareSideHeader from './CompareSideHeader';

interface Props {
  /** Стороны сравнения; первая — базовая. */
  sides: CompareSide[];
  /** Версии моделей, параллельно sides. */
  versions: MLModelVersion[];
  /** Метрики моделей, параллельно sides (для строки о сохранённых весах); null — нет данных. */
  metricsBySide: Array<ModelMetrics | null>;
}

interface MetaRow {
  key: string;
  label: React.ReactNode;
  /** Ячейки значений, параллельно sides. */
  cells: React.ReactNode[];
  differs: boolean;
}

const shortId = (id: string) => (id.length > 10 ? `${id.slice(0, 8)}…` : id);

// Значение листа train_params: отсутствующее — прочерк, строки — как есть,
// остальное (числа, булевы, null, массивы) — JSON: null остаётся однозначным
// «null», а не неотличимым от строки 'null'.
const formatParam = (value: unknown) =>
  value === undefined
    ? '—'
    : typeof value === 'string'
      ? value
      : JSON.stringify(value);

// Конфиг обучения динамический и вложенный: разворачиваем в плоские
// dot-пути (optimizer.lr), чтобы diff шёл по отдельным значениям,
// а не по JSON-блобу целиком. Массивы — листья.
const flattenParams = (
  params: Record<string, unknown>,
  prefix = '',
  out: Map<string, unknown> = new Map(),
): Map<string, unknown> => {
  for (const [key, value] of Object.entries(params)) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      flattenParams(value as Record<string, unknown>, path, out);
    } else {
      out.set(path, value);
    }
  }
  return out;
};

const allEqual = (values: string[]) => values.every((v) => v === values[0]);

// Длиннее — значение сворачивается до превью (списки аугментаций и подобные
// параметры конфига могут занимать сотни символов и ломать таблицу).
const PARAM_PREVIEW_LIMIT = 80;

/** Значение параметра: длинное сворачивается до превью с кнопкой раскрытия. */
const ParamValue: React.FC<{ text: string }> = ({ text }) => {
  const [expanded, setExpanded] = useState(false);
  if (text.length <= PARAM_PREVIEW_LIMIT) return <code>{text}</code>;
  return (
    <span className="mcmp-param-long">
      <code className={expanded ? 'mcmp-param-long--expanded' : undefined}>
        {expanded ? text : `${text.slice(0, PARAM_PREVIEW_LIMIT)}…`}
      </code>
      <button
        type="button"
        className="mcmp-param-toggle"
        onClick={() => setExpanded((e) => !e)}
      >
        {expanded ? 'свернуть' : 'показать полностью'}
      </button>
    </span>
  );
};

/**
 * Diff метаданных и конфигов обучения сравниваемых моделей: строки
 * с различающимися значениями подсвечены; по умолчанию показываются
 * только различия — в духе W&B Run Comparer. Рендерится независимо
 * от наличия метрик.
 */
const ModelCompareMetaDiff: React.FC<Props> = ({ sides, versions, metricsBySide }) => {
  const [diffOnly, setDiffOnly] = useState(true);
  // Имена датасетов и их версий (модель хранит только id). Ошибка загрузки
  // не критична — ячейки откатываются на сокращённые id.
  const [datasets, setDatasets] = useState<Map<string, Dataset>>(new Map());

  const datasetIds = Array.from(new Set(versions.map((v) => v.dataset_id)))
    .sort()
    .join('|');

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      datasetIds.split('|').map((id) =>
        datasetService.getDataset(id).catch(() => null),
      ),
    ).then((results) => {
      if (cancelled) return;
      setDatasets(new Map(
        results.filter((d): d is Dataset => d !== null).map((d) => [d.id, d]),
      ));
    });
    return () => { cancelled = true; };
  }, [datasetIds]);

  const classesCell = (version: MLModelVersion) => (
    <Tooltip content={version.classes.join(', ') || undefined}>
      <span>{version.classes.length}</span>
    </Tooltip>
  );
  const datasetCell = (version: MLModelVersion) => {
    const dataset = datasets.get(version.dataset_id);
    return (
      <Tooltip content={version.dataset_id}>
        <Link to={`/datasets/${version.dataset_id}`}>
          {dataset?.name ?? shortId(version.dataset_id)}
        </Link>
      </Tooltip>
    );
  };
  const datasetVersionCell = (version: MLModelVersion) => {
    const name = datasets
      .get(version.dataset_id)
      ?.versions.find((v) => v.id === version.dataset_version_id)?.name;
    return (
      <Tooltip content={version.dataset_version_id}>
        <span>{name ?? shortId(version.dataset_version_id)}</span>
      </Tooltip>
    );
  };
  const framework = (v: MLModelVersion) =>
    v.framework ? `${v.framework}${v.framework_version ? ` ${v.framework_version}` : ''}` : '—';
  const classesKey = (v: MLModelVersion) => [...v.classes].sort().join(' ');

  // «Сохранённые веса»: эпоха чекпоинта и early-stop-метрика. Общее число
  // эпох выводим из длины кривых train/val (структуру конфига не хардкодим).
  // checkpoint.value == null — улучшений не фиксировалось, веса финальной эпохи.
  const checkpointText = (metrics: ModelMetrics | null) => {
    const ckpt = metrics?.checkpoint;
    if (!ckpt) return 'неизвестно';
    const totalEpochs = totalEpochsOf(metrics);
    const ofTotal = totalEpochs ? ` из ${totalEpochs}` : '';
    if (ckpt.value == null) {
      return `веса финальной эпохи ${ckpt.epoch}${ofTotal} (улучшений ${ckpt.metric} на валидации не было)`;
    }
    return `веса с эпохи ${ckpt.epoch}${ofTotal} — лучший ${ckpt.metric} на валидационной выборке`;
  };
  const checkpointCells = metricsBySide.map(checkpointText);

  const metaRows: MetaRow[] = [
    {
      key: 'model_type',
      label: 'Тип модели',
      cells: versions.map((v) => v.model_type),
      differs: !allEqual(versions.map((v) => v.model_type)),
    },
    {
      key: 'framework',
      label: 'Фреймворк',
      cells: versions.map(framework),
      differs: !allEqual(versions.map(framework)),
    },
    {
      key: 'checkpoint',
      label: (
        <Tooltip content="Какая эпоха обучения ушла в реестр моделей и тестировалась: trainer сохраняет веса эпохи с лучшим значением early-stop-метрики (по умолчанию — loss) на валидационной выборке">
          <span>Сохранённые веса</span>
        </Tooltip>
      ),
      cells: checkpointCells,
      differs: !allEqual(checkpointCells),
    },
    {
      key: 'classes',
      label: 'Классы',
      cells: versions.map(classesCell),
      differs: !allEqual(versions.map(classesKey)),
    },
    {
      key: 'dataset',
      label: 'Датасет',
      cells: versions.map(datasetCell),
      differs: !allEqual(versions.map((v) => v.dataset_id)),
    },
    {
      key: 'dataset_version',
      label: 'Версия датасета',
      cells: versions.map(datasetVersionCell),
      differs: !allEqual(versions.map((v) => v.dataset_version_id)),
    },
    {
      key: 'created_at',
      label: 'Создана',
      cells: versions.map((v) => formatDateParts(v.created_at).date),
      // Даты создания различаются почти всегда — не подсвечиваем как diff.
      differs: false,
    },
  ];

  // Объединение листовых путей конфигов всех моделей: параметра может
  // не быть у части сторон — тогда «—».
  const paramsBySide = versions.map((v) => flattenParams(v.train_params));
  const paramKeys = Array.from(
    new Set(paramsBySide.flatMap((params) => [...params.keys()])),
  ).sort();
  const paramRows: MetaRow[] = paramKeys.map((key) => {
    const texts = paramsBySide.map((params) =>
      formatParam(params.has(key) ? params.get(key) : undefined),
    );
    return {
      key: `param:${key}`,
      label: <code>{key}</code>,
      cells: texts.map((text, index) => (
        <ParamValue key={sides[index].id} text={text} />
      )),
      differs: !allEqual(texts),
    };
  });

  const renderRows = (rows: MetaRow[]) =>
    rows
      .filter((row) => !diffOnly || row.differs)
      .map((row) => (
        <tr key={row.key} className={row.differs ? 'mcmp-meta-row--diff' : undefined}>
          <th>{row.label}</th>
          {row.cells.map((cell, index) => (
            <td key={sides[index].id}>{cell}</td>
          ))}
        </tr>
      ));

  const visibleMeta = renderRows(metaRows);
  const visibleParams = renderRows(paramRows);

  return (
    <section className="detail-section">
      <div className="mcmp-meta-head">
        <p className="metrics-split-title">
          <i className={`fas ${ICONS.trainingParams}`}></i> Конфигурация моделей
          <InfoHint text="Метаданные и параметры обучения сравниваемых моделей. По умолчанию показаны только различия — полный конфиг открывается переключателем справа. Конфиг динамический: набор параметров у моделей может отличаться, отсутствующее значение показано прочерком." />
        </p>
        <label className="mcmp-meta-toggle">
          <input
            type="checkbox"
            checked={diffOnly}
            onChange={(e) => setDiffOnly(e.target.checked)}
          />
          Только различия
        </label>
      </div>
      <p className="mcmp-section-desc">
        Метаданные и параметры обучения. Подсвеченные строки различаются —
        именно эти отличия объясняют разницу в метриках; длинные значения
        свёрнуты до превью.
      </p>
      {visibleMeta.length === 0 && visibleParams.length === 0 ? (
        <div className="metrics-charts-placeholder">
          <i className={`fas ${ICONS.success}`}></i> Конфигурации моделей идентичны.
        </div>
      ) : (
        <div className="class-report-table-wrap mcmp-table-wrap mcmp-meta-table-wrap">
          <table className="class-report-table mcmp-meta-table">
            <thead>
              <tr>
                <th>Параметр</th>
                {sides.map((side) => (
                  <CompareSideHeader key={side.id} side={side} />
                ))}
              </tr>
            </thead>
            <tbody>
              {visibleMeta}
              {visibleParams.length > 0 && (
                <tr className="mcmp-meta-subhead">
                  <th colSpan={sides.length + 1}>Параметры обучения</th>
                </tr>
              )}
              {visibleParams}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};

export default ModelCompareMetaDiff;
