# Этап 1: Загрузка LFS-файлов
FROM alpine/git AS lfs
RUN apk add --no-cache git-lfs
WORKDIR /repo
RUN git clone --branch main --depth 1 https://github.com/annashesta/ml_service_hw_1_with_docker_image.git .

RUN git lfs install && git lfs pull

# Этап 2: Основной образ
FROM python:3.12-slim
WORKDIR /app

# Копируем LFS-файлы из первого этапа
COPY --from=lfs /repo/model/catboost_model.cbm /app/model/
COPY --from=lfs /repo/model/threshold.json /app/model/
COPY --from=lfs /repo/train_data/train.csv /app/train_data/
COPY --from=lfs /repo/model/categorical_features.json /app/model/

# Копируем остальные файлы напрямую
COPY requirements.txt .
COPY config.yaml .
COPY src/ /app/src/
COPY app/app.py /app/app.py

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

# Настройка прав
RUN mkdir -p /app/input /app/output /app/logs && \
    useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/logs

USER appuser
CMD ["python", "app.py"]
