import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

print("🍄 開始重建瑪利歐知識庫...")

# ==========================================
# 1. 準備知識庫資料 (這裡放你想讓 AI 知道的事情)
# ==========================================
documents = [
    "路易吉 (Luigi) 是瑪利歐的雙胞胎弟弟，穿著綠色的衣服，跳得比瑪利歐高。",
    "碧姬公主 (Princess Peach) 經常被庫巴綁架，瑪利歐必須去城堡救她。",
    "瑪利歐的主要敵人是庫巴 (Bowser)，他是一隻會噴火的巨大烏龜。",
    "瑪利歐吃下火之花 (Fire Flower) 後可以丟出火球攻擊敵人。"
]

# 給每筆資料一個獨一無二的 ID
ids = ["doc_1", "doc_2", "doc_3", "doc_4"]

# ==========================================
# 2. 清理舊環境 (防呆機制)
# ==========================================
db_path = "./mario_db_local"
if os.path.exists(db_path):
    print(f"🧹 發現舊的資料庫資料夾 ({db_path})，正在為你清空...")
    shutil.rmtree(db_path)

# ==========================================
# 3. 初始化「輕量級」Embedding 模型 (解決 512MB RAM 限制的關鍵)
# ==========================================
print("📦 正在下載/載入輕量級模型 (all-MiniLM-L6-v2)...")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# 4. 建立與寫入 ChromaDB 資料庫
# ==========================================
print("🔨 正在建立新的向量資料庫...")
chroma_client = chromadb.PersistentClient(path=db_path)

# 建立 Collection
collection = chroma_client.create_collection(
    name="mario_knowledge",
    embedding_function=emb_fn
)

# 將文本寫入
print("💾 正在將文本轉成向量並存入資料庫...")
collection.add(
    documents=documents,
    ids=ids
)

print("✅ 大功告成！全新的 mario_db_local 資料夾已產生！")