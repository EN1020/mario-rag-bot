#把模型下載下來存到 Docker 的快取區
from chromadb.utils import embedding_functions

print("📦 正在預先下載模型，請稍候...")

# 這裡的模型名稱必須跟你 rag_bot.py 裡用的一模一樣！
embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

print("✅ 模型下載完成！已存入 Docker 快取。")