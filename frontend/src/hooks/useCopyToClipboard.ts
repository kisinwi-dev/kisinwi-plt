import { useCallback } from 'react';
import { useNotification } from '../contexts/NotificationContext';

/**
 * Хук копирования текста в буфер обмена с уведомлением об успехе.
 * Возвращает стабильный колбэк (text) => void.
 * e.stopPropagation() при необходимости вызывать на месте (хук работает только с текстом).
 */
export const useCopyToClipboard = (successMessage = 'ID скопирован'): ((text: string) => void) => {
  const { showNotification } = useNotification();
  return useCallback(
    (text: string) => {
      navigator.clipboard.writeText(text);
      showNotification(successMessage, 'success');
    },
    [showNotification, successMessage],
  );
};
