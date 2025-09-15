from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import json

with open("knowledge.json", "r", encoding="utf-8") as f:
    data = json.load(f)

items = data["items"]

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for item in items:
        document = Document(
            page_content=item["title"] + " " + item["text"],
            metadata = {"id": item["id"]},
            id = item["id"]
        )
        ids.append(item["id"])
        documents.append(document)
        
vector_store = Chroma(
    collection_name="moozi_docs",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)