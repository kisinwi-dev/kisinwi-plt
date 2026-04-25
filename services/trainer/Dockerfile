# Выбираем в зависимости от верии cuda на железе (https://hub.docker.com/r/pytorch/pytorch)
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Копируем requirements
COPY requirements.txt .

# Установка стандартных библиотек
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Устанавливаем torchvision под CUDA 12.8
RUN pip install --no-cache-dir --break-system-packages torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Копируем код проекта
COPY . /app

# Сервис использует порт 6200
EXPOSE 6200

CMD ["python", "main.py"]