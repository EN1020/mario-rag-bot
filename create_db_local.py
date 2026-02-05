import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. 設定本地端 Embedding 模型
# ==========================================
# 這裡我們使用 ChromaDB 內建的整合功能
# 它會自動幫我們下載模型並執行
print("正在載入本地端模型 (第一次執行會下載約 400MB，請稍候)...")

emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2" 
)

# ==========================================
# 2. 準備資料
# ==========================================
documents = [
    "超級瑪利歐兄弟 (Super Mario Bros.) 是由任天堂開發的平台遊戲。",
    "瑪利歐的主要敵人是庫巴 (Bowser)，他是一隻會噴火的巨大烏龜。",
    "在遊戲中，瑪利歐吃下超級蘑菇 (Super Mushroom) 後會變大，能破壞磚塊。",
    "瑪利歐吃下火之花 (Fire Flower) 後可以丟出火球攻擊敵人。",
    "碧姬公主 (Princess Peach) 經常被庫巴綁架，瑪利歐必須去城堡救她。",
    "路易吉 (Luigi) 是瑪利歐的雙胞胎弟弟，穿著綠色的衣服，跳得比瑪利歐高。",
    "無敵星 (Super Star) 可以讓瑪利歐在短時間內變成無敵狀態，撞倒所有敵人。"
]
ids = [str(i) for i in range(len(documents))]

# ==========================================
# 3. 初始化並寫入資料庫
# ==========================================
# 注意：這次我們換個資料夾名字 "mario_db_local"，避免跟舊的混淆
client = chromadb.PersistentClient(path="./mario_db_local")

# 重要：我們必須告訴資料庫「以後用這個函式來處理向量」
collection = client.get_or_create_collection(
    name="mario_knowledge",
    embedding_function=emb_fn 
)

print("正在寫入資料庫...")
collection.upsert(
    documents=documents,
    ids=ids
)

print(f"✅ 成功寫入 {len(documents)} 筆資料到本地資料庫！")