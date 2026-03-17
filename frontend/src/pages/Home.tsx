import React from 'react';
import './Home.css'; // если используешь обычный CSS, или styles, если модули

const Home: React.FC = () => {
  return (
    <div className="home">
      <header className="hero">
        <h1>KiSinWi</h1>
        <p className="tagline">
          Платформа, которая упрощает жизнь ML‑инженеров и открывает мир 
          машинного обучения для всех, кто далёк от кода.
        </p>
      </header>

      <section className="mission">
        <h2>О проекте</h2>
        <p>
          KiSinWi создан, чтобы снять барьеры в работе с моделями компьютерного зрения. 
          Мы хотим, чтобы инженеры тратили меньше времени на рутину, а новички могли 
          быстро получить рабочую модель, не погружаясь в дебри алгоритмов.
        </p>
      </section>

      <section className="features">
        <h2>Основные возможности</h2>
        <div className="feature-grid">
          <div className="feature-card">
            <h3>📁 Работа с датасетами</h3>
            <p>
              Загружай, удаляй и версионируй датасеты. 
              Сейчас поддерживается <strong>Image Classification</strong>.
            </p>
          </div>
          <div className="feature-card">
            <h3>⚙️ Запуск обучения</h3>
            <p>
              Выбирай модель и параметры, запускай эксперименты 
              прямо из интерфейса. Всё прозрачно и настраиваемо.
            </p>
          </div>
          <div className="feature-card">
            <h3>🤖 Обучение с агентами</h3>
            <p>
              Агенты сами подбирают гиперпараметры, анализируют результаты 
              и предлагают улучшения — машинное обучение становится доступнее.
            </p>
          </div>
        </div>
      </section>

      <footer className="status">
        <p>
          <em>На данный момент в фокусе — задачи классификации изображений, 
          но мы активно расширяем функциональность.</em>
        </p>
      </footer>
    </div>
  );
};

export default Home;