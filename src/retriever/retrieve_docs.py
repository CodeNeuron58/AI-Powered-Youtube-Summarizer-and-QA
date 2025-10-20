from typing import Tuple, List

from src.preprocessing.embeddings import embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def retrieve_documents(docs: List[Document], k: int = 5) -> Tuple[FAISS, object]:
    """Build a FAISS vector store from documents and return (vector_store, retriever).

    Args:
        docs: list of LangChain Document objects
        k: top-k to use for retriever by default

    Returns:
        (vector_store, retriever)
    """
    emb = embeddings()
    vector_store = FAISS.from_documents(docs, emb)
    retriever = vector_store.as_retriever(search_kwargs={"k": k}, search_type="similarity")

    return vector_store, retriever