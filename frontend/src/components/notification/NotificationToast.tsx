import React from 'react';
// Импортируем хук для доступа к контексту уведомлений.
import { useNotification } from '../../contexts/NotificationContext';
import { ICONS } from '../../constants/icons';
// Подключаем стили для компонента уведомлений.
import './NotificationToast.css';

/**
 * Возвращает класс иконки Font Awesome в зависимости от типа уведомления.
 * @param type - тип уведомления ('error', 'success', 'warning', 'info').
 * @returns класс иконки Font Awesome.
 */
const getIcon = (type: string) => {
  switch (type) {
    case 'error': return ICONS.error;
    case 'success': return ICONS.success;
    case 'warning': return ICONS.warning;
    case 'info': return ICONS.info;
    default: return ICONS.info;
  }
};

/**
 * Компонент для отображения всплывающих уведомлений (тостов).
 * Использует контекст уведомлений для получения активных уведомлений
 * и функции их удаления. Рендерит каждый тост в виде карточки с иконкой,
 * текстом и кнопкой закрытия.
 */
const NotificationToast: React.FC = () => {
  // Получаем из контекста массив уведомлений и функцию удаления по id.
  const { notifications, removeNotification } = useNotification();

  // Если нет уведомлений, ничего не рендерим (экономим место в DOM).
  if (notifications.length === 0) return null;

  return (
    // Контейнер для всех тостов. Фиксированное позиционирование, выравнивание по правому краю.
    <div className="notification-container">
      {notifications.map(n => (
        // Каждый тост – отдельный блок. Класс формируется на основе типа (например, notification-error).
        <div key={n.id} className={`notification notification-${n.type}`}>
          <div className="notification-content">
            {/* Иконка, соответствующая типу */}
            <span className="notification-icon"><i className={`fas ${getIcon(n.type)}`}></i></span>
            {/* Текст сообщения */}
            <span className="notification-message">{n.message}</span>
          </div>
          {/* Кнопка закрытия – при клике удаляем уведомление по id */}
          <button className="notification-close" onClick={() => removeNotification(n.id)}>
            <i className={`fas ${ICONS.close}`}></i>
          </button>
        </div>
      ))}
    </div>
  );
};

export default NotificationToast;