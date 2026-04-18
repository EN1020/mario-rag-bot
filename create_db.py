import chromadb
from google import genai
import os
import shutil
import getpass # 🛡️ 新增這個安全套件

print("🍄 開始建立【純 API 雲端版】瑪利歐知識庫...")

# 1. 取得 API Key (使用 getpass 隱藏輸入)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # ⚠️ 安全機制：輸入時畫面不會顯示任何字，貼上後直接按 Enter 即可
    api_key = getpass.getpass("🔑 請貼上你的 GOOGLE_API_KEY: ")

client = genai.Client(api_key=api_key)

# 2. 準備知識庫資料
documents = [
    "路易吉 (Luigi) 是瑪利歐的雙胞胎弟弟，穿著綠色的衣服，跳得比瑪利歐高。",
    "碧姬公主 (Princess Peach) 經常被庫巴綁架，瑪利歐必須去城堡救她。",
    "瑪利歐的主要敵人是庫巴 (Bowser)，他是一隻會噴火的巨大烏龜。",
    "瑪利歐吃下火之花 (Fire Flower) 後可以丟出火球攻擊敵人。"
]
ids = ["doc_1", "doc_2", "doc_3", "doc_4"]

# 3. 清理舊環境
db_path = "./mario_db_local"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# 4. 透過 Google API 將文字轉成向量
print("☁️ 正在呼叫 Google API 計算向量...")
embeddings = []
for doc in documents:
    response = client.models.embed_content(
        model="gemini-embedding-001", # 🔥 更新為最新模型
        contents=doc
    )
    embeddings.append(response.embeddings[0].values)

# 5. 寫入 ChromaDB
print("💾 正在寫入資料庫...")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.create_collection(name="mario_knowledge")

collection.add(
    documents=documents,
    embeddings=embeddings, 
    ids=ids
)

print("✅ 大功告成！全新的 API 版資料庫已產生！")