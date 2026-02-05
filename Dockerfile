# 1. 基底映像檔：我們使用官方瘦身版的 Python 3.10
FROM python:3.10-slim

# 2. 設定工作目錄：這就像是 CD 到某個資料夾
WORKDIR /app

# 3. 安裝系統依賴 (選用，但為了避免 sentence-transformers 缺件，裝一下比較保險)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製需求清單並安裝
# (先複製清單是為了 Docker Cache 機制，加速之後的 Build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
# 複製預載腳本並執行
COPY preload_model.py .
RUN python preload_model.py
# 複製程式碼與資料庫
# 我們把目前的 rag_bot.py 和已經建好的 mario_db_local 資料夾都複製進去
COPY rag_bot.py .
COPY mario_db_local ./mario_db_local


# 1. 告訴 Docker 這個程式會用 8501 Port (Streamlit 預設埠)
EXPOSE 8501

# 2. 修改啟動指令
# --server.address=0.0.0.0 是關鍵！這讓外部(你的 Windows)可以連進來
CMD ["streamlit", "run", "rag_bot.py", "--server.address=0.0.0.0"]