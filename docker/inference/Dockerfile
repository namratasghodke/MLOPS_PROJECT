FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Increase timeout to 120s, retry 5 times
RUN pip install --default-timeout=120 --retries=5 -r requirements.txt

COPY src/inference/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
