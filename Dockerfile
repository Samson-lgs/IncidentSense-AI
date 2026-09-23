FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/generate_demo_data.py && python scripts/train_model.py

EXPOSE 8000
CMD ["uvicorn", "incidentsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
