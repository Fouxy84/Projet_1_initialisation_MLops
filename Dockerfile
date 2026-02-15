FROM python:3.10-slim

# 🔧 Installer dépendance système requise par LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Code + modèle
COPY api ./api
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
