CREATE TYPE task_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

CREATE TABLE IF NOT EXISTS train_models_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL,

    -- Используемая модель
    model_id VARCHAR(255) NOT NULL,
    
    -- Агенты и id диалога
    discussion_id VARCHAR(255),
    agent_respons_ids JSONB DEFAULT '[]',

    -- Статус
    status task_status NOT NULL DEFAULT 'pending',
    status_info TEXT,
    error_message TEXT,
    
    -- Время
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Индексы
CREATE INDEX idx_tasks_status ON train_models_tasks(status);
CREATE INDEX idx_tasks_created ON train_models_tasks(created_at DESC);

-- Авто обновление времени

CREATE OR REPLACE FUNCTION handle_task_timestamps()
RETURNS TRIGGER AS $$
BEGIN
    -- Обновляем updated_at всегда
    NEW.updated_at = CURRENT_TIMESTAMP;
    
    -- Если статус меняется с pending на processing
    IF OLD.status = 'pending' AND NEW.status = 'processing' THEN
        NEW.started_at = CURRENT_TIMESTAMP;
    END IF;
    
    -- Если статус меняется на completed
    IF NEW.status IN ('completed') THEN
        NEW.completed_at = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER handle_train_models_tasks_timestamps
    BEFORE UPDATE ON train_models_tasks
    FOR EACH ROW
    EXECUTE FUNCTION handle_task_timestamps();