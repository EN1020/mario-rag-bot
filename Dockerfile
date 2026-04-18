FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_bot.py .
COPY mario_db_local ./mario_db_local

EXPOSE 8501

CMD ["streamlit", "run", "rag_bot.py", "--server.address=0.0.0.0"]