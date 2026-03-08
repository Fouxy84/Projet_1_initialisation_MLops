FROM python:3.10-slim

# 🔧 Installer dépendance système requise par LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*


ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/app/notebook/mlruns
WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + modèle

COPY . .

#port API
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
