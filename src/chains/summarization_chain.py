from src.promt.summarization_promt import load_summary_promt

def create_summarization_chain(llm):
    summary_promt = load_summary_promt()
    chain = summary_promt | llm
    return chain