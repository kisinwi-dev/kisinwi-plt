CREATE TYPE task_status AS ENUM (
    'pending',
    'processing',
    'completed',
    'failed'
);

CREATE TABLE IF NOT EXISTS train_models_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Информация по используемой модели
    model_id VARCHAR(255) NOT NULL,
    
    -- Информация по агентам
    discussion_id VARCHAR(255),
    agent_respons_ids JSONB DEFAULT '[]',
    
    -- Статус
    status task_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    
    -- Время
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ
);

-- Минимум индексов
CREATE INDEX idx_tasks_status ON train_models_tasks(status);
CREATE INDEX idx_tasks_created ON train_models_tasks(created_at DESC);

-- Авто обновление времени

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_train_models_tasks_updated_at
    BEFORE UPDATE ON train_models_tasks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column()

