# Пояснения к package.json

## dependencies (работают в браузере)
- **react**: библиотека для интерфейсов, компоненты, useState, useEffect
- **react-dom**: рендеринг React в HTML
- **react-router-dom**: переключение страниц (роутинг)
- **antd**: готовые компоненты Ant Design (кнопки, инпуты, таблицы)
- **axios**: HTTP запросы к бекенду
- **react-hook-form**: управление формами и валидация

## devDependencies (только для разработки)
- **typescript**: компилятор TS в JS
- **vite**: быстрый сервер и сборщик
- **@vitejs/plugin-react**: поддержка React в Vite
- **eslint**: проверка ошибок в коде
- **@types/react**: подсказки типов для React в VS Code

## scripts (команды)
- `npm run dev` - запуск сервера разработки
- `npm run build` - сборка для продакшена
- `npm run lint` - проверка кода
- `npm run preview` - просмотр собранного проекта