from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_transcript(transcript, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(texts=[transcript])
    return docs
