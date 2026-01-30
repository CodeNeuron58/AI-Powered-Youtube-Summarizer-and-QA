from typing import Optional, List
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.transcript.get_youtube_transcript import get_transcript
from src.preprocessing.transcipt_splitter import split_transcript
from src.retriever.retrieve_docs import retrieve_documents
from src.promt.qa_promt import load_qa_promt

try:
    from src.chains.LLM import load_llm
except Exception:
    load_llm = None

from langchain.chains import RetrievalQA

app = FastAPI(title="YouTube QA & Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadRequest(BaseModel):
    video_url: str


class QARequest(BaseModel):
    question: str


class SummarizeRequest(BaseModel):
    # When true, summarize the full transcript stored in memory. If false, summarize top-k relevant to `prompt`.
    full: bool = True
    prompt: Optional[str] = None


# In-memory state (simple)
state = {
    "plain_text": None,
    "docs": None,
    "vector_store": None,
    "retriever": None,
}


@app.get("/")
def root():
    return {"message": "YouTube QA & Summarizer API", "status": "ok"}


@app.post("/load_transcript")
def load_transcript(req: LoadRequest):
    try:
        txt = get_transcript(req.video_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch transcript: {e}")

    # split and build vector store
    docs = split_transcript(txt)
    vector_store, retriever = retrieve_documents(docs)

    state["plain_text"] = txt
    state["docs"] = docs
    state["vector_store"] = vector_store
    state["retriever"] = retriever

    return {"message": "transcript loaded", "chunks": len(docs)}


@app.post("/qa")
def qa(req: QARequest):
    if state.get("retriever") is None:
        raise HTTPException(status_code=400, detail="No retriever available. Load a transcript first.")

    qa_prompt = load_qa_promt()
    retriever = state["retriever"]

    # build chain
    try:
        llm = load_llm() if load_llm else None
        qa_chain = RetrievalQA.from_llm(llm, retriever=retriever, prompt=qa_prompt, return_source_documents=True)
    except Exception:
        qa_chain = RetrievalQA.from_llm(load_llm() if load_llm else None, retriever=retriever, prompt=qa_prompt)

    try:
        res = qa_chain({"query": req.question})
        answer = res.get("result") or res.get("output_text") or res.get("answer") or res
        sources = res.get("source_documents") if isinstance(res, dict) else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {e}")

    return {"answer": str(answer), "sources": [s.page_content for s in (sources or [])]}


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if not state.get("plain_text"):
        raise HTTPException(status_code=400, detail="No transcript loaded. Call /load_transcript first.")

    if req.full:
        transcript_text = state["plain_text"]
    else:
        if not req.prompt:
            raise HTTPException(status_code=400, detail="When full=false you must provide a prompt to retrieve top-k documents.")
        retriever = state.get("retriever")
        if retriever is None:
            raise HTTPException(status_code=400, detail="No retriever available. Load transcript first.")
        top_docs = retriever.get_relevant_documents(req.prompt)
        transcript_text = "\n\n".join(d.page_content for d in top_docs)

    # summarization prompt
    from langchain.prompts import PromptTemplate

    SUMMARY_TEMPLATE = """
You are an expert summarizer. Read the following transcript and produce a complete, comprehensive summary that captures all main points, key arguments, and conclusions. The summary should be a coherent, self-contained text and include the most important details from the transcript.

Transcript:
{transcript}

Please produce the full summary below:
"""

    summary_prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["transcript"])
    filled = summary_prompt.format(transcript=transcript_text)

    try:
        llm = load_llm() if load_llm else None
        if llm is None:
            raise RuntimeError("LLM not configured on server. Set COHERE_API_KEY in .env and ensure load_llm is available.")
        try:
            summary = llm.invoke(filled)
        except Exception as e_invoke:
            # best effort
            raise RuntimeError(f"LLM invocation failed: {e_invoke}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"summary": str(summary)}
