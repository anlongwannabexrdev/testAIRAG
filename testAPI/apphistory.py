from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders import PDFPlumberLoader

from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_pdf(file_path):
    pages = convert_from_path(file_path, poppler_path=r"C:\poppler-24.07.0\Library\bin")
    docs = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang='eng+vie')
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": i+1, "source": file_path}))
    return docs

app = Flask(__name__)

chat_history = []

folder_path = "db"

cached_llm = Ollama(model="llama3.2")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. Just answer based on the context below.
    Always answer in Vietnamese (include if you do not have any answer). 
    If you do not have an answer from the provided information, say so in Vietnamese. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", "You are a helpful AI in searching documents for users. Always answer in Vietnamese."),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generation a search query to lookup in order to get information relevant to the conversation. Always answer in Vietnamese.",
            ),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=cached_llm, retriever=retriever, prompt=retriever_prompt
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    # chain = create_retrieval_chain(retriever, document_chain)

    retrieval_chain = create_retrieval_chain(
        # retriever,
        history_aware_retriever,
        document_chain,
    )

    # result = chain.invoke({"input": query})
    result = retrieval_chain.invoke({"input": query})

    if not result["context"]:
        return {"answer": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong tài liệu.", "sources": []}

    print(result["answer"])
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    #Load File
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    #If dont have text -> fallback to ocr
    if not docs:
        #loader = UnstructuredPDFLoader(save_file, strategy="ocr_only")
        docs = ocr_pdf(save_file)
        print(f"After OCR docs len={len(docs)}")

    #Split to chunks
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    if not chunks:
        return {
            "status": "fail",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": 0,
            "error": "Không trích xuất được nội dung từ PDF."
        }
    
    #Save to Chroma
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
