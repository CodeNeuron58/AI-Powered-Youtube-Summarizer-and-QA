from typing import Tuple, List

from src.preprocessing.embeddings import embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def retrieve_documents(docs: List[Document], k: int = 5, embeddings_model=None) -> Tuple[FAISS, object]:
    """Build a FAISS vector store from documents and return (vector_store, retriever).

    Args:
        docs: list of LangChain Document objects
        k: top-k to use for retriever by default
        embeddings_model: Optional, pre-loaded embeddings model. If None, loads new one.

    Returns:
        (vector_store, retriever)
    """
    emb = embeddings_model if embeddings_model is not None else embeddings()
    vector_store = FAISS.from_documents(docs, emb)
    retriever = vector_store.as_retriever(search_kwargs={"k": k}, search_type="similarity")

    return vector_store, retriever