from src.promt.qa_promt import load_qa_promt
from src.chains.LLM import load_llm
from langchain.chains import RetrievalQA
from src.retriever.retrieve_docs import retrieve_documents

def create_qa_chain(retriver):
    qa_promt = load_qa_promt()
    llm = load_llm()
    chain = RetrievalQA.from_llm(llm=llm, prompt=qa_promt, retriever=retriver)
    return chain


