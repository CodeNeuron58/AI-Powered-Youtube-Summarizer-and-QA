from langchain_huggingface import HuggingFaceEmbeddings

def embeddings():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
