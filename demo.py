import faiss
import numpy as np
import json
from ollama import Client

# 1. Khởi tạo client Ollama
ollama = Client()

# 2. Load file .json
with open("knowledge.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [item["text"] for item in data["items"]]
print("Loaded", len(documents), "documents from knowledge.json")

# 3. Hàm tạo embedding với nomic-embed-text
def get_embedding(text):
    res = ollama.embed(model="nomic-embed-text", input=text)
    return np.array(res["embeddings"][0], dtype="float32")

# 4. Tạo index FAISS
dimension = len(get_embedding("test"))  # kích thước vector từ model
index = faiss.IndexFlatL2(dimension)

for doc in documents:
    emb = get_embedding(doc)
    index.add(np.array([emb]))

print("✅ Đã build xong FAISS index.")

# Hàm rerank bằng LLM
def rerank(query, candidates):
    context = "\n".join([f"-{c}" for c in candidates])
    prompt= f"""
    Câu hỏi: {query}

    Các đoạn trích dưới đây được lấy từ knowledge.json:
    {context}

    Nhiệm vụ: Hãy CHỌN ra những đoạn LIÊN QUAN NHẤT để trả lời câu hỏi.
    Trả về nguyên văn đoạn văn bản phù hợp nhất (1–2 đoạn). 
    Nếu không có đoạn nào liên quan, trả về: "Không tìm thấy".
    """

    res = ollama.chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": prompt}]
    )
    return res["message"]["content"]

# 5. Hàm truy vấn RAG
def rag_query(query, top_k=2):
    q_emb = get_embedding(query)
    D, I = index.search(np.array([q_emb]), top_k)
    candidates = [documents[i] for i in I[0]]

    # Dùng LLM lọc lại
    filtered_context = rerank(query, candidates)

    final_prompt = f"""
    Bạn là một trợ lý chỉ được phép trả lời dựa trên THÔNG TIN THAM CHIẾU bên dưới.
    Nếu không có thông tin liên quan, hãy trả lời: "Không tìm thấy trong dữ liệu."

    Câu hỏi: {query}
    Thông tin tham chiếu:
    {filtered_context}

    Hãy trả lời dựa trên thông tin trên.
    """
    response = ollama.chat(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return response["message"]["content"]

while True:
    query = input("Nhập prompt ('exit' để thoát): ")
    if query.lower() == "exit":
        break

    answer = rag_query(query)
    print("💡 Answer:", answer)
