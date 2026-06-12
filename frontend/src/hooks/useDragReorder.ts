import React, { useEffect, useMemo, useState } from 'react';

/**
 * Перетаскивание элементов сетки/списка для смены их порядка (HTML5 DnD).
 *
 * Перетаскивание начинается только с «ручки» (handleProps) — содержимое
 * элемента (hover-тултипы графиков, выделение текста) остаётся рабочим.
 * Порядок применяется на лету при наведении на другой элемент; при заданном
 * storageKey сохраняется в localStorage. Ключи, которых нет в сохранённом
 * порядке (новые метрики), добавляются в конец в исходном порядке.
 */
export function useDragReorder(keys: string[], storageKey?: string) {
  const [order, setOrder] = useState<string[]>(() => {
    if (!storageKey) return [];
    try {
      const raw = localStorage.getItem(storageKey);
      return raw ? (JSON.parse(raw) as string[]) : [];
    } catch {
      return [];
    }
  });
  // Элемент «взведён» нажатием на ручку: только он получает draggable,
  // иначе перетаскивание стартовало бы с любой точки блока.
  const [armedKey, setArmedKey] = useState<string | null>(null);
  const [draggedKey, setDraggedKey] = useState<string | null>(null);

  const orderedKeys = useMemo(() => {
    const known = order.filter((k) => keys.includes(k));
    return [...known, ...keys.filter((k) => !known.includes(k))];
  }, [order, keys]);

  useEffect(() => {
    if (!storageKey || order.length === 0) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(order));
    } catch {
      // localStorage недоступен (приватный режим) — порядок живёт до перезагрузки.
    }
  }, [order, storageKey]);

  // Отпустил ручку без перетаскивания — снимаем draggable с блока.
  useEffect(() => {
    if (armedKey === null) return;
    const disarm = () => setArmedKey(null);
    window.addEventListener('mouseup', disarm);
    return () => window.removeEventListener('mouseup', disarm);
  }, [armedKey]);

  const moveTo = (targetKey: string) => {
    if (!draggedKey || draggedKey === targetKey) return;
    const next = orderedKeys.filter((k) => k !== draggedKey);
    next.splice(next.indexOf(targetKey), 0, draggedKey);
    if (next.every((k, i) => k === orderedKeys[i])) return;
    setOrder(next);
  };

  const handleProps = (key: string) => ({
    onMouseDown: () => setArmedKey(key),
  });

  const itemProps = (key: string) => ({
    draggable: armedKey === key,
    onDragStart: (e: React.DragEvent) => {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', key);
      setDraggedKey(key);
    },
    onDragOver: (e: React.DragEvent) => {
      if (draggedKey === null) return; // чужой drag (файл, текст) — игнорируем
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      moveTo(key);
    },
    onDrop: (e: React.DragEvent) => e.preventDefault(),
    onDragEnd: () => {
      setDraggedKey(null);
      setArmedKey(null);
    },
  });

  return { orderedKeys, draggedKey, handleProps, itemProps };
}
