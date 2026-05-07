-- ============================================================================
-- Таблица для информации о статусах
-- ============================================================================

CREATE TABLE IF NOT EXISTS task_statuses (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL UNIQUE,
    description TEXT NOT NULL
);

INSERT INTO task_statuses (status, description) VALUES
    ('waiting', 'Ожидает начало выполнения'),
    ('running', 'Выполняется'),
    ('completed', 'Завершено'),
    ('failed', 'Завершена с ошибкой');

-- ============================================================================
-- Таблица для информации о задачах тренировок
-- ============================================================================

CREATE TABLE IF NOT EXISTS train_models_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL,

    -- Используемая модель
    model_id VARCHAR(255) NOT NULL,
    
    -- Агенты и id диалога
    discussion_id VARCHAR(255),
    agent_respons_ids JSONB DEFAULT '[]',

    -- Статус
    status_id INTEGER NOT NULL DEFAULT 1,
    percentages INTEGER NOT NULL DEFAULT 0,
    status_info TEXT,
    error_message TEXT,
    
    -- Время
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,

    -- связь с таблицей статусов
    CONSTRAINT fk_task_status FOREIGN KEY (status_id) 
        REFERENCES task_statuses(id)
);

-- Индексы
CREATE INDEX idx_tasks_status_id ON train_models_tasks(status_id);
CREATE INDEX idx_tasks_created ON train_models_tasks(created_at DESC);

-- Авто обновление времени

CREATE OR REPLACE FUNCTION handle_task_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    -- Обновляем updated_at всегда
    NEW.updated_at = CURRENT_TIMESTAMP;
    
    -- Если статус меняется на статус обучения
    IF NEW.status_id = 2 AND OLD.status_id != 2 THEN
        NEW.started_at = CURRENT_TIMESTAMP;
    END IF;

    -- Если задача успешно завершена
    IF NEW.status_id = 3 AND OLD.status_id != 3 THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;
    
    -- Если задача закончена с ошибкой
    IF NEW.status_id = 4 AND OLD.status_id != 4 THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER handle_train_models_tasks_timestamps
    BEFORE UPDATE ON train_models_tasks
    FOR EACH ROW
    EXECUTE FUNCTION handle_task_timestamps();