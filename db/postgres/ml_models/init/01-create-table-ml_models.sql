-- ENUM для статуса модели
CREATE TYPE ml_model_status_enum AS ENUM (
    'draft',    -- модель ещё не обучена или не готова
    'training', -- в процессе обучения
    'completed' -- готова к использованию
);

-- Таблица ML моделей (разбирается только вариант с классификацией изображений)
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    model_type VARCHAR(100) NOT NULL,
    description TEXT,
    classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Датасет для которого создана модель
    dataset_id UUID NOT NULL,
    dataset_version_id VARCHAR(20) NOT NULL, 

    -- Framework для работы с моделью
    framework VARCHAR(50),
    framework_version VARCHAR(20),

    -- Путь до весов
    storage_path TEXT,

    -- Параметры обучения модели
    train_params JSONB NOT NULL,

    -- Провкрка верно введённых данных
    -- уникальность имени и версии
    CONSTRAINT unique_model_name_version UNIQUE (name, version),
    -- атрибут с количеством классов не должен быть пустым
    CONSTRAINT check_classes_not_empty CHECK (jsonb_array_length(classes) > 0)
);

-- Индексы
CREATE INDEX idx_ml_models_type ON ml_models(model_type);
CREATE INDEX idx_ml_models_name ON ml_models(name);
CREATE INDEX idx_ml_models_dataset ON ml_models(dataset_id, dataset_version_id);
CREATE INDEX idx_ml_models_name_version ON ml_models(name, version DESC)

