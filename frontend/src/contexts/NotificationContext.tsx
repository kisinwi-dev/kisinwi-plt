import React, { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';

// Определяем возможные типы уведомлений. Это union строковых литералов.
// Используется для стилизации и определения иконки в компоненте Toast.
type NotificationType = 'error' | 'success' | 'info' | 'warning';

// Интерфейс одного уведомления:
// - id – уникальный идентификатор (для удаления из массива и ключа в списке)
// - message – текст сообщения
// - type – тип (для визуального оформления)
interface Notification {
  id: string;
  message: string;
  type: NotificationType;
}

// Тип для значения, которое будет храниться в контексте:
// - notifications – массив активных уведомлений
// - showNotification – функция для добавления нового уведомления
// - removeNotification – функция для удаления уведомления по id
interface NotificationContextType {
  notifications: Notification[];
  showNotification: (message: string, type?: NotificationType) => void;
  removeNotification: (id: string) => void;
}

// Создаём сам контекст с начальным значением undefined.
// TypeScript будет требовать проверку на undefined при использовании useContext,
// чтобы гарантировать, что контекст используется внутри провайдера.
const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

// Компонент-провайдер. Он оборачивает часть приложения, где нужны уведомления.
// Принимает children (ReactNode) – вложенные компоненты.
export const NotificationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Состояние: массив уведомлений. Изначально пустой.
  const [notifications, setNotifications] = useState<Notification[]>([]);

  // Функция для показа уведомления.
  // Принимает текст и необязательный тип (по умолчанию 'error').
  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const showNotification = useCallback((message: string, type: NotificationType = 'error') => {
    const id = Date.now().toString() + Math.random().toString(36).slice(2, 7);
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => removeNotification(id), 5000);
  }, [removeNotification]);

  // Провайдер передаёт дочерним компонентам объект с текущими уведомлениями и функциями.
  return (
    <NotificationContext.Provider value={{ notifications, showNotification, removeNotification }}>
      {children}
    </NotificationContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useNotification = () => {
  const context = useContext(NotificationContext);
  // Если контекст undefined – значит, хук вызвали вне провайдера. Выбрасываем ошибку.
  if (!context) throw new Error('useNotification must be used within NotificationProvider');
  return context;
};