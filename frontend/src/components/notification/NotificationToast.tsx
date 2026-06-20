import React from 'react';
import { useNotification } from '../../contexts/NotificationContext';
import { ICONS } from '../../constants/icons';
import './NotificationToast.css';

const getIcon = (type: string) => {
  switch (type) {
    case 'error': return ICONS.error;
    case 'success': return ICONS.success;
    case 'warning': return ICONS.warning;
    case 'info': return ICONS.info;
    default: return ICONS.info;
  }
};

const NotificationToast: React.FC = () => {
  const { notifications, removeNotification } = useNotification();

  if (notifications.length === 0) return null;

  return (
    <div className="notification-container">
      {notifications.map(n => (
        <div key={n.id} className={`notification notification-${n.type}`}>
          <div className="notification-content">
            <span className="notification-icon"><i className={`fas ${getIcon(n.type)}`}></i></span>
            <span className="notification-message">{n.message}</span>
          </div>
          <button className="notification-close" onClick={() => removeNotification(n.id)}>
            <i className={`fas ${ICONS.close}`}></i>
          </button>
        </div>
      ))}
    </div>
  );
};

export default NotificationToast;
