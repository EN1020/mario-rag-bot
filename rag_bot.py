import chromadb
from google import genai
import os
import streamlit as st

st.set_page_config(page_title="🍄 瑪利歐 AI 專家", page_icon="🍄")
st.title("🍄 瑪利歐遊戲知識問答 RAG (雲端極速版)")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ 錯誤：找不到 GOOGLE_API_KEY 環境變數！")
    st.stop()

@st.cache_resource
def load_resources():
    print("正在連線資料庫...")
    client = genai.Client(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path="./mario_db_local")
    # 注意：這裡不用再設定 embedding_function 了
    collection = chroma_client.get_collection(name="mario_knowledge")
    return client, collection

client, collection = load_resources()

def get_answer(question):
    try:
        # Step 1: 將使用者的問題轉成向量 (呼叫 Google API)
        emb_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )
        query_embedding = emb_response.embeddings[0].values

        # Step 2: 拿向量去資料庫搜尋最相似的資料
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=2
        )
        retrieved_text = "\n".join(results['documents'][0])
        
        # Step 3: 生成最終回答
        prompt = f"""
        你是一個專業的瑪利歐遊戲專家。請根據以下的【參考資料】回答使用者的問題。
        如果資料裡找不到答案，請說「不好意思，我的知識庫裡沒有相關資訊」。
        
        【參考資料】：
        {retrieved_text}
        
        【使用者問題】：
        {question}
        """
        
        # 請確保這裡是你能用的回答模型
        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=prompt
        )
        return response.text, retrieved_text
    except Exception as e:
        return f"❌ 錯誤: {e}", ""

# --- 網頁互動區 ---
user_input = st.chat_input("請輸入關於瑪利歐的問題...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🍄 呼叫 Google 雲端思考中..."):
            answer, ref_data = get_answer(user_input)
            st.write(answer)
            
            with st.expander("查看 AI 參考了哪些資料"):
                st.info(ref_data)