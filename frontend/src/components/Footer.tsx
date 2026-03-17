import React from 'react';
import './Footer.css';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-section">
          <h4 className="footer-title">KiSinWi</h4>
          <p className="footer-description">
            Платформа для ML-инженеров и всех, кто хочет быстро создавать модели компьютерного зрения.
          </p>
        </div>


        <div className="footer-section">
          <h4 className="footer-title">Связь</h4>
          <div className="social-links">
            <a 
              href="https://telegram.me/andrySin#" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="social-link"
              title="Telegram"
            >
              <i className="fab fa-telegram"></i>
            </a>
            <a 
              href="https://github.com/kisinwi-dev/kisinwi-plt" 
              target="_blank" 
              rel="noopener noreferrer" 
              className="social-link"
              title="GitHub"
            >
              <i className="fab fa-github"></i>
            </a>
          </div>
          <button className="back-to-top" onClick={scrollToTop}>
            <i className="fas fa-arrow-up"></i> Наверх
          </button>
        </div>
      </div>

      <div className="footer-bottom">
        <p className="footer-copyright">
          © {currentYear} KiSinWi. Все права защищены.
        </p>
      </div>
    </footer>
  );
};

export default Footer;