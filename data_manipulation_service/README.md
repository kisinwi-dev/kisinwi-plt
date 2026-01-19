# Data Manipulation Service

[Русский](##русский) | [English](##english)

## Русский <a name="русский"></a>

Сервис для работы с данными.  
- Код лежит в `src/`.
- Данные находятся в `datasets/` и подключаются через Docker volume.

---

### Структура проекта

```
data_manipulation_service/
├─ src/
│ ├─ main.py
│ ├─ ...
│ └─ datasets/ <- для локальных эксперементов без контейнеризации создайте папку
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ README.md
```

---

### 1️⃣ Локальный запуск (без Docker)

1. Установи зависимости:

```bash
pip install -r requirements.txt
```

2. Запуск сервиса:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 1907
```

3. Проверка:
```
http://localhost:1907/docs
```


### 2️⃣ Запуск через Docker

1. Сборка образа

```bash
docker build -t data_manipulation_service .
```

2. Запуск контейнера с volume для данных

```
docker run -d -p 1907:1907 \
  -v $(pwd)/src/datasets:/app/datasets \
  -e DATA_PATH=/app/datasets \
  --name data_manipulation_service \
  data_manipulation_service
```

Пояснения:
* `-p 1907:1907` - проброс порта.
* `-v $(pwd)/src/datasets:/app/datasets` - локальная папка с данными подключается как volume.
* `-e DATA_PATH=/app/datasets` - переменная окружения для пути к данным внутри контейнера.

3. Проверка:

```bash
http://localhost:1907/docs
```

4. Остановка контейнера

```
docker stop data_manipulation_service
docker rm data_manipulation_service
```

### 3️⃣ Примечания

* Код и данные разделены для масштабируемости.
* Volume позволяет обновлять данные без пересборки Docker-образа.


## English <a name="english"></a>

Service for data manipulation.
- Code is located in `src/`.
- Data is in `datasets/` and mounted via Docker volume.

---

### Project structure

```
data_manipulation_service/
├─ src/
│ ├─ main.py
│ ├─ ...
│ └─ datasets/ <- for local experiments without Docker
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
└─ README.md
```

---

### 1️⃣ Local run (without Docker)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the service:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 1907
```

3. Test:
```bash
http://localhost:1907/docs
```


### 2️⃣ Run via Docker

1. Build the image:

```bash
docker build -t data_manipulation_service .
```

2. Run container with volume:

```
docker run -d -p 1907:1907 \
  -v $(pwd)/src/datasets:/app/datasets \
  -e DATA_PATH=/app/datasets \
  --name data_manipulation_service \
  data_manipulation_service
```

Explanation:
* `-p 1907:1907` - port mapping.
* `-v $(pwd)/src/datasets:/app/datasets` - mount local datasets folder as volume.
* `-e DATA_PATH=/app/datasets` — environment variable for dataset path inside the container.

3. Test:

```bash
http://localhost:1907/docs
```

4. Stop container:

```
docker stop data_manipulation_service
docker rm data_manipulation_service
```

### 3️⃣ Notes:

* Code and data are separated for scalability.
* Volume allows updating datasets without rebuilding the Docker image.