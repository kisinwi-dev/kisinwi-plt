// Единый реестр иконок интерфейса (Font Awesome 6, CDN подключён в index.html).
// Значения хранятся без префикса `fas`/`fab` — как в theme/themes.ts и SYSTEM_ICONS.
// Использование в JSX: <i className={`fas ${ICONS.delete}`} />
//   brand-иконки (секция «бренды») идут с префиксом `fab`: <i className={`fab ${ICONS.brandGithub}`} />
//   loading-спиннер: <i className={`fas ${ICONS.loading} fa-spin`} />
// Ключи — это семантические РОЛИ, поэтому совпадение значений у разных ролей
// (например classes/tags = fa-tags) допустимо и намеренно.

export const ICONS = {
  // --- действия ---
  add: 'fa-plus',
  delete: 'fa-trash',          // удаление сущности (датасет, версия, модель, дискуссия)
  close: 'fa-xmark',           // закрытие/сброс/удаление элемента списка/файла
  search: 'fa-search',
  copy: 'fa-copy',
  filter: 'fa-filter',
  back: 'fa-arrow-left',
  toTop: 'fa-arrow-up',
  external: 'fa-arrow-up-right-from-square',
  download: 'fa-download',
  star: 'fa-star',             // версия по умолчанию
  play: 'fa-play',             // запуск pipeline
  expand: 'fa-chevron-down',
  collapse: 'fa-chevron-up',
  pagePrev: 'fa-chevron-left',
  pageNext: 'fa-chevron-right',
  selected: 'fa-check',        // выбранный пункт в списке/дропдауне
  listView: 'fa-list',         // переключатель «плоский список»
  groupedView: 'fa-layer-group', // переключатель «группировка по моделям»

  // --- статусы ---
  loading: 'fa-spinner',       // + fa-spin
  info: 'fa-circle-info',
  success: 'fa-circle-check',
  warning: 'fa-triangle-exclamation',
  error: 'fa-circle-exclamation',
  notFound: 'fa-triangle-exclamation',
  question: 'fa-circle-question',
  empty: 'fa-box-open',        // пустой список результатов

  // --- сущности ---
  dataset: 'fa-database',
  datasetType: 'fa-shapes',
  model: 'fa-cube',
  framework: 'fa-layer-group', // ML-фреймворк (PyTorch/TF и т.п.)
  version: 'fa-code-branch',
  agent: 'fa-robot',
  agentsGroup: 'fa-users',
  agentModel: 'fa-brain',       // LLM-модель агента
  discussion: 'fa-comments',    // обсуждение / ответы агентов
  noMessages: 'fa-comment-slash',
  pipeline: 'fa-diagram-project',
  task: 'fa-list-check',
  tools: 'fa-wrench',
  trainingParams: 'fa-sliders',
  metrics: 'fa-chart-line',
  datasetStats: 'fa-chart-bar', // статистика splits версии датасета
  report: 'fa-file-lines',
  weights: 'fa-file-arrow-down',
  file: 'fa-file',

  // --- данные / мета ---
  id: 'fa-hashtag',
  tag: 'fa-tag',                // тип/задача (одиночный ярлык)
  tags: 'fa-tags',             // классы / множество ярлыков
  classes: 'fa-tags',
  samples: 'fa-images',
  link: 'fa-link',
  description: 'fa-align-left',
  taskTarget: 'fa-bullseye',   // поле «Задача» датасета
  dateCreated: 'fa-calendar-alt',
  dateUpdated: 'fa-rotate',
  dateFinished: 'fa-flag-checkered',
  duration: 'fa-clock',
  iteration: 'fa-rotate',
  history: 'fa-clock-rotate-left',

  // --- файлы (FileUploader) ---
  upload: 'fa-cloud-arrow-up',
  fileArchive: 'fa-file-zipper',
  fileImage: 'fa-file-image',
  fileTable: 'fa-file-csv',
  fileText: 'fa-file-lines',

  // --- темы оформления (переключатель) ---
  themeDark: 'fa-moon',
  themeLight: 'fa-sun',
  themeMidnight: 'fa-cloud-moon',

  // --- бренды (использование с префиксом `fab`, а не `fas`) ---
  brandTelegram: 'fa-telegram',
  brandGithub: 'fa-github',
} as const;

export type IconKey = keyof typeof ICONS;
