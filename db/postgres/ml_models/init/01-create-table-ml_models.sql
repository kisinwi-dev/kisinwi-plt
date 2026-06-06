-- Таблица статсусов моделей
CREATE TABLE IF NOT EXISTS ml_model_statuses (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL UNIQUE,
    description TEXT NOT NULL
);

INSERT INTO ml_model_statuses (status, description) VALUES
    ('draft', 'Не обучена'),
    ('training', 'В процессе обучения'),
    ('completed', 'Обучена');

-- ============================================================================
-- Таблица ML моделей (разбирается только вариант с классификацией изображений)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    status_id INTEGER NOT NULL DEFAULT 1,
    description TEXT,
    metrics_result TEXT DEFAULT 'No info',
    classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Датасет для которого создана модель (UUID, 36 символов)
    dataset_id VARCHAR(36) NOT NULL,
    dataset_version_id VARCHAR(36) NOT NULL,

    -- Framework для работы с моделью
    framework VARCHAR(50),
    framework_version VARCHAR(20),

    -- Параметры обучения модели
    train_params JSONB NOT NULL,

    -- Провкрка верно введённых данных
    -- уникальность имени и версии
    CONSTRAINT unique_model_name_version UNIQUE (name, version),
    -- атрибут с количеством классов не должен быть пустым
    CONSTRAINT check_classes_not_empty CHECK (jsonb_array_length(classes) > 0),
    -- связь с таблицей статусов
    CONSTRAINT fk_ml_models_status FOREIGN KEY (status_id) 
        REFERENCES ml_model_statuses(id)
);

-- Индексы
CREATE INDEX idx_ml_models_type ON ml_models(model_type);
CREATE INDEX idx_ml_models_name ON ml_models(name);
CREATE INDEX idx_ml_models_dataset ON ml_models(dataset_id, dataset_version_id);
CREATE INDEX idx_ml_models_name_version ON ml_models(name, version DESC);

-- ============================================================================
-- Таблица для информации о файлах
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_model_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_ml_model_files_model 
        FOREIGN KEY (model_id) REFERENCES ml_models(id) ON DELETE CASCADE
);

-- Индексы
CREATE INDEX idx_ml_model_files_model_id ON ml_model_files(model_id);