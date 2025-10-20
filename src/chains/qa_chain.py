from src.promt.qa_promt import load_qa_promt


def create_qa_chain(llm):
    qa_promt = load_qa_promt()
    chain = qa_promt | llm
    return chain


