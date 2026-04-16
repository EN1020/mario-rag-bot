import chromadb
from chromadb.utils import embedding_functions
from google import genai
import os
import streamlit as st  # 1. 引入 streamlit

# 設定頁面標題
st.set_page_config(page_title="🍄 瑪利歐 AI 專家", page_icon="🍄")
st.title("🍄 瑪利歐遊戲知識問答 RAG")

# 檢查 API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ 錯誤：找不到 GOOGLE_API_KEY 環境變數！")
    st.stop()

# ==========================================
# 2. 初始化 (加上快取，避免每次對話都重跑)
# ==========================================
@st.cache_resource
def load_resources():
    print("正在載入模型與資料庫...")
    # 初始化 Google Client
    client = genai.Client(api_key=api_key)
    
    # 初始化 Embedding
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 連線資料庫 (注意：Docker 內的路徑)
    chroma_client = chromadb.PersistentClient(path="./mario_db_local")
    collection = chroma_client.get_collection(
        name="mario_knowledge",
        embedding_function=emb_fn
    )
    return client, collection

# 載入資源
client, collection = load_resources()

# ==========================================
# 3. 定義回答邏輯 (跟原本一樣)
# ==========================================
def get_answer(question):
    # Step 1: 檢索
    results = collection.query(query_texts=[question], n_results=2)
    
    # 這裡做個小改動：如果距離太遠，直接回傳找不到 (這是一個保險)
    # 不過為了展示 RAG 效果，我們先維持原本的 Prompt 機制
    retrieved_text = "\n".join(results['documents'][0])
    
    # Step 2: 生成
    prompt = f"""
    你是一個專業的瑪利歐遊戲專家。請根據以下的【參考資料】回答使用者的問題。
    如果資料裡找不到答案，請說「不好意思，我的知識庫裡沒有相關資訊」。
    
    【參考資料】：
    {retrieved_text}
    
    【使用者問題】：
    {question}
    """
    try:
        # 請記得確認這裡的模型名稱是你帳號能用的 (例如 gemini-2.0-flash-exp)
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt
        )
        return response.text, retrieved_text
    except Exception as e:
        return f"❌ 錯誤: {e}", ""

# ==========================================
# 4. 網頁互動區 (取代原本的 while 迴圈)
# ==========================================

# 建立一個聊天輸入框
user_input = st.chat_input("請輸入關於瑪利歐的問題...")

if user_input:
    # 1. 顯示使用者的訊息
    with st.chat_message("user"):
        st.write(user_input)

    # 2. 顯示 AI 思考過程與回答
    with st.chat_message("assistant"):
        with st.spinner("🍄 瑪利歐正在翻閱百科全書..."):
            answer, ref_data = get_answer(user_input)
            st.write(answer)
            
            # (選用) 顯示它參考了什麼資料，讓這看起來更像除錯工具
            with st.expander("查看 AI 參考了哪些資料"):
                st.info(ref_data)
