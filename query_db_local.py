import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. 設定跟 create 時一模一樣的模型
# ==========================================
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# ==========================================
# 2. 連線
# ==========================================
client = chromadb.PersistentClient(path="./mario_db_local")
collection = client.get_collection(
    name="mario_knowledge",
    embedding_function=emb_fn 
)

def search_knowledge_base(query_text):
    print(f"\n使用者問題: {query_text}")
    
    # 直接搜尋！Chroma 會自動呼叫 emb_fn 幫你的問題轉向量
    results = collection.query(
        query_texts=[query_text], # 注意：這裡直接傳文字，不用自己 embed 了
        n_results=2,
        include=['documents', 'distances']
    )
    
    best_distance = results['distances'][0][0]
    
    # 本地模型的距離計算方式可能略有不同 (通常是 L2 distance 或 Cosine)
    # 這裡的閥值我們先抓寬一點來觀察
    threshold = 15.0 # L2 distance 的數字會比較大，不像 Cosine 是 0~1
    
    if best_distance > threshold:
        print(f"⚠️ 距離過遠 ({best_distance:.4f})，系統判定不知道。")
    else:
        print(f"✅ 找到答案 (距離 {best_distance:.4f})：")
        print(f"內容: {results['documents'][0][0]}")

# ==========================================
# 測試
# ==========================================
search_knowledge_base("瑪利歐的弟弟是誰？")
search_knowledge_base("薩爾達公主住在哪裡？")