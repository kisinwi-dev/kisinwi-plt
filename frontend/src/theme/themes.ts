// Реестр тем оформления и хелперы переключения.
// Сами палитры объявлены в styles/theme.css блоками [data-theme="<id>"].

export type ThemeId = 'dark' | 'light' | 'midnight';

export interface ThemeOption {
  id: ThemeId;
  label: string;
  /* FontAwesome-иконка для переключателя */
  icon: string;
}

export const THEMES: ThemeOption[] = [
  { id: 'dark', label: 'Тёмная', icon: 'fa-moon' },
  { id: 'light', label: 'Светлая', icon: 'fa-sun' },
  { id: 'midnight', label: 'Полночь', icon: 'fa-cloud-moon' },
];

const STORAGE_KEY = 'kisinwi-theme';
const DEFAULT_THEME: ThemeId = 'dark';

const isThemeId = (value: string | null): value is ThemeId =>
  !!value && THEMES.some((t) => t.id === value);

/** Прочитать сохранённую тему (или дефолт). */
export const getStoredTheme = (): ThemeId => {
  const stored = localStorage.getItem(STORAGE_KEY);
  return isThemeId(stored) ? stored : DEFAULT_THEME;
};

/** Применить тему к документу и сохранить выбор. */
export const applyTheme = (theme: ThemeId): void => {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(STORAGE_KEY, theme);
};
