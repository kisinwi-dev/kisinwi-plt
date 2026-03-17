import React, { createContext, useContext, useState } from 'react';
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
  const showNotification = (message: string, type: NotificationType = 'error') => {
    // Генерируем уникальный идентификатор:
    // - Date.now() даёт миллисекунды
    // - случайная строка из Math.random (36-ричная система, удаляем первые 2 символа)
    const id = Date.now().toString() + Math.random().toString(36).substr(2, 5);

    // Добавляем новое уведомление в конец массива.
    setNotifications(prev => [...prev, { id, message, type }]);

    // Автоматически удаляем уведомление через 5 секунд.
    setTimeout(() => removeNotification(id), 5000);
  };

  // Функция удаления уведомления по id.
  // Фильтрует массив, оставляя все элементы, кроме того, чей id совпадает.
  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  // Провайдер передаёт дочерним компонентам объект с текущими уведомлениями и функциями.
  return (
    <NotificationContext.Provider value={{ notifications, showNotification, removeNotification }}>
      {children}
    </NotificationContext.Provider>
  );
};

// Хук для удобного доступа к контексту уведомлений.
// Должен использоваться только внутри компонентов, обёрнутых в NotificationProvider.
export const useNotification = () => {
  const context = useContext(NotificationContext);
  // Если контекст undefined – значит, хук вызвали вне провайдера. Выбрасываем ошибку.
  if (!context) throw new Error('useNotification must be used within NotificationProvider');
  return context;
};