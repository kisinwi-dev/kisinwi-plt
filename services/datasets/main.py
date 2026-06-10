import uvicorn
from fastapi import FastAPI

from app.api.routers import api_routers
from fastapi.middleware.cors import CORSMiddleware

openapi_tags = [
    {
        "name": "Upload",
        "description": "Загрузка архивов с данными во временное хранилище. "
                       "Первый шаг перед созданием датасета или новой версии.",
    },
    {
        "name": "Datasets",
        "description": "CRUD датасетов: создание из загруженных данных, "
                       "получение метаданных, смена стандартной версии, удаление.",
    },
    {
        "name": "Versions",
        "description": "CRUD версий датасета: создание из загруженных данных, "
                       "получение метаданных, удаление.",
    },
    {
        "name": "Version Stats",
        "description": "Статистика версии по сплитам: количество изображений, "
                       "баланс и распределение классов, размеры и форматы изображений.",
    },
    {
        "name": "Version Compare",
        "description": "Сравнение двух версий датасета: дельты количества, "
                       "drift-метрики распределений (JS divergence, PSI), "
                       "баланс, размеры, по-файловый diff.",
    },
    {
        "name": "Info",
        "description": "Служебные эндпоинты: healthcheck сервиса и зависимостей.",
    },
]

app = FastAPI(
    title="Datasets Service",
    version="0.1.0",
    description="""
Сервис отвечает за управление датасетами и получение информации о них.
""",
    openapi_tags=openapi_tags,
)

app.include_router(
    api_routers
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6500,
        reload=True,
    )
