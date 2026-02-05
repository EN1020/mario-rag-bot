from google import genai

# 設定 API Key
GOOGLE_API_KEY = "AIzaSyCypVbNWdBRK_dtv8nvuS1wN1PAFKyOfPU"
client = genai.Client(api_key=GOOGLE_API_KEY)

print("正在查詢你的帳號可用的【生成】模型清單...\n")

try:
    # 列出所有模型
    for m in client.models.list():
        # 這次我們找支援 "generateContent" 的模型
        if "generateContent" in m.supported_generation_methods:
            print(f"✅ 可用模型: {m.name}")
            
except Exception as e:
    print(f"❌ 查詢失敗: {e}")

print("\n查詢結束。")