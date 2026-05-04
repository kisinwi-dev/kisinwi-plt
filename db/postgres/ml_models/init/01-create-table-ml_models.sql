-- Таблица моделей для классификации изображений
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    description TEXT,
    classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Параметры обучения модели
    train_params JSONB NOT NULL,
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Индексы
CREATE INDEX idx_ml_models_type ON ml_models(model_type);
CREATE INDEX idx_ml_models_name ON ml_models(name);
