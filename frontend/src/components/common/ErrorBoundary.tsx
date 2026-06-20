import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

// Ловит неперехваченные ошибки рендера дочерних компонентов, чтобы падение
// одной страницы не превращало всё приложение в белый экран. Header/Footer/тосты
// живут выше по дереву и переживают сбой — пользователь может уйти на другой роут.
class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Необработанная ошибка рендера:', error, info.componentStack);
  }

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div style={{ padding: '3rem 1rem', textAlign: 'center' }}>
        <h2>Что-то пошло не так</h2>
        <p>Страница не смогла отобразиться. Попробуйте перезагрузить.</p>
        <button type="button" onClick={() => window.location.reload()}>
          Перезагрузить
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
