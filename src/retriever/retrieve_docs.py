from src.preprocessing.embeddings import embeddings
from langchain_community.vectorstores import FAISS


def retrieve_documents(docs, query, k=5):
    db = FAISS.from_documents(docs, embeddings())
    return db.similarity_search(query, k=k)

