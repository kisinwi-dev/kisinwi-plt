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
-- Таблица моделей (родительская сущность: имя + описание)
-- ============================================================================

CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT unique_model_name UNIQUE (name)
);

-- ============================================================================
-- Таблица версий модели (разбирается только вариант с классификацией изображений)
-- ============================================================================

CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    model_type VARCHAR(100) NOT NULL,
    status_id INTEGER NOT NULL DEFAULT 1,
    metrics_report TEXT DEFAULT 'No info',
    classes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Датасет для которого создана версия (UUID, 36 символов)
    dataset_id VARCHAR(36) NOT NULL,
    dataset_version_id VARCHAR(36) NOT NULL,

    -- Framework для работы с моделью
    framework VARCHAR(50),
    framework_version VARCHAR(20),

    -- Параметры обучения модели
    train_params JSONB NOT NULL,

    -- уникальность номера версии внутри модели
    CONSTRAINT unique_model_version UNIQUE (model_id, version),
    -- атрибут с количеством классов не должен быть пустым
    CONSTRAINT check_classes_not_empty CHECK (jsonb_array_length(classes) > 0),
    -- связь с таблицей статусов
    CONSTRAINT fk_model_versions_status FOREIGN KEY (status_id)
        REFERENCES ml_model_statuses(id),
    -- связь с родительской моделью
    CONSTRAINT fk_model_versions_model FOREIGN KEY (model_id)
        REFERENCES models(id) ON DELETE CASCADE
);

-- Индексы
CREATE INDEX idx_model_versions_model_id ON model_versions(model_id, version DESC);
CREATE INDEX idx_model_versions_dataset ON model_versions(dataset_id, dataset_version_id);
CREATE INDEX idx_model_versions_status ON model_versions(status_id);

-- ============================================================================
-- Таблица для информации о файлах
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_model_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_ml_model_files_version
        FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

-- Индексы
CREATE INDEX idx_ml_model_files_version_id ON ml_model_files(version_id);
