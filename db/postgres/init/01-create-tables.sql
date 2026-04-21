-- Создание таблиц
\set ON_ERROR_STOP on

-- Создание enum тип для статуса задач
CREATE TYPE task_status AS ENUM (
    'pending',
    'in_progress',
    'completed',
    'failed'
);

-- основная информация о задачах
CREATE TABLE IF NOT EXISTS training_tasks (
    -- Задача
    id_task UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_name VARCHAR(255),
    -- Время
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    end_at TIMESTAMPTZ,
    -- Статус
    status task_status NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    -- Параметры обучения
    params JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- изменения в процессе выполнения
CREATE TABLE IF NOT EXISTS training_task_logs (
    -- Индентификаторы
    id SERIAL PRIMARY KEY,
    id_task UUID NOT NULL REFERENCES training_tasks(id_task) ON DELETE CASCADE,
    -- Прогресс на момент записи лога
    stage_progress INTEGER NOT NULL,
    -- Описание этапа
    info TEXT NOT NULL,
    -- Время
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- Индексы для оптимизации запросов

-- Для быстрого поиска по статусу и id
CREATE INDEX IF NOT EXISTS idx_tasks_status ON training_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_task_name ON training_tasks(id_task);

-- Индексы для таблицы логов
CREATE INDEX IF NOT EXISTS idx_logs_task ON training_task_logs(id_task);

-- Комментарии к таблицам и колонкам

COMMENT ON TABLE training_tasks IS 'Основная информация о задачах тренировки моделей';
COMMENT ON COLUMN training_tasks.id_task IS 'Уникальный идентификатор задачи (UUID)';
COMMENT ON COLUMN training_tasks.task_name IS 'Имя задачи';
COMMENT ON COLUMN training_tasks.status IS 'Статус задачи: pending, in_progress, completed, failed';
COMMENT ON COLUMN training_tasks.progress IS 'Прогресс выполнения';
COMMENT ON COLUMN training_tasks.params IS 'JSON параметры для запуска обучения (model_name, batch_size, epochs и т.д.)';

COMMENT ON TABLE training_task_logs IS 'Логи изменений в процессе выполнения задач';
COMMENT ON COLUMN training_task_logs.stage_progress IS 'Прогресс на момент записи лога';
COMMENT ON COLUMN training_task_logs.info IS 'Сообщение о текущем этапе выполнения';
COMMENT ON COLUMN training_task_logs.created_at IS 'Время создания лога';

-- Тригеры

-- Автоматическое обновление updated_at

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_training_tasks_updated_at 
    BEFORE UPDATE ON training_tasks
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Автоматическое обновление прогресса при добавлении логов


CREATE OR REPLACE FUNCTION update_task_progress()
RETURNS TRIGGER AS $$
BEGIN
    -- Обновляем прогресс в основной таблице
    UPDATE training_tasks 
    SET progress = NEW.stage_progress
    WHERE id_task = NEW.id_task;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_progress_on_log 
    AFTER INSERT ON training_task_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_task_progress();

-- Лог результаов действий

DO $$
BEGIN
    RAISE NOTICE '✅ Таблицы созданы:';
    RAISE NOTICE '   - training_tasks (таблица задач)';
    RAISE NOTICE '   - training_task_logs (`список` изменений)';
    RAISE NOTICE '✅ Индексы созданы';
    RAISE NOTICE '✅ Триггеры созданы';
END $$;