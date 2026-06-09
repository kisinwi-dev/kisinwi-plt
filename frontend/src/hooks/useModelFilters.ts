import { useState, useCallback } from 'react';

export interface ModelFilters {
  name: string;
  status: string;
  dataset: string;
}

const EMPTY_FILTERS: ModelFilters = { name: '', status: '', dataset: '' };

/**
 * Состояние фильтров списка моделей вместе с пагинацией.
 * Любое изменение фильтра сбрасывает страницу на первую.
 */
export const useModelFilters = () => {
  const [filters, setFilters] = useState<ModelFilters>(EMPTY_FILTERS);
  const [offset, setOffset] = useState(0);

  const setFilter = useCallback((key: keyof ModelFilters, value: string) => {
    setOffset(0);
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  const resetPage = useCallback(() => setOffset(0), []);

  return { filters, offset, setFilter, setOffset, resetPage };
};
