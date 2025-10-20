import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.preprocessing.transcipt_splitter import split_transcript
from src.retriever.retrieve_docs import retrieve_documents
from src.promt.qa_promt import load_qa_promt

# Try to import LLM factory (optional in repo)
try:
	from src.chains.LLM import load_llm
except Exception:
	load_llm = None

st.set_page_config(page_title="YouTube Summarizer & QA", layout="wide")

st.title("YouTube Summarizer & QA")

with st.sidebar:
	st.header("Settings")
	cohere_key = st.text_input("COHERE_API_KEY (or set in .env)", type="password")
	k = st.number_input("retriever k (top documents)", min_value=1, max_value=20, value=5)

cohere_key = cohere_key or os.getenv("COHERE_API_KEY")
if not cohere_key:
	st.warning("Set COHERE_API_KEY in the sidebar or in .env to use Cohere LLMs. You can still test retrieval without it.")


def fetch_transcript(video_id_or_url: str) -> str:
	vid = video_id_or_url.strip().split("v=")[-1].split("&")[0]
	fetched = YouTubeTranscriptApi().fetch(vid, languages=["en"])
	raw = fetched.to_raw_data()
	return " ".join(chunk["text"] for chunk in raw)


def build_vector_store_from_text(text: str, k_val: int = 5):
	# split into docs using the project's splitter
	docs = split_transcript(text)
	vector_store, retriever = retrieve_documents(docs, k=int(k_val))
	return vector_store, retriever, docs


# Session state keys
if "vector_store" not in st.session_state:
	st.session_state["vector_store"] = None
if "retriever" not in st.session_state:
	st.session_state["retriever"] = None
if "plain_text" not in st.session_state:
	st.session_state["plain_text"] = None
if "docs" not in st.session_state:
	st.session_state["docs"] = None

st.subheader("Load YouTube transcript")
video_input = st.text_input("YouTube video URL or ID", value="")
load_btn = st.button("Load transcript & build vector store")

if load_btn and video_input:
	try:
		with st.spinner("Fetching transcript..."):
			plain_text = fetch_transcript(video_input)
		with st.spinner("Building vector store..."):
			vector_store, retriever, docs = build_vector_store_from_text(plain_text, k_val=k)

		st.session_state["vector_store"] = vector_store
		st.session_state["retriever"] = retriever
		st.session_state["plain_text"] = plain_text
		st.session_state["docs"] = docs
		st.success(f"Built vector store with {len(docs)} chunks")
	except Exception as e:
		st.error(f"Failed to load transcript or build store: {e}")

if st.session_state.get("vector_store") is not None:
	st.sidebar.markdown(f"**Chunks:** {len(st.session_state.get('docs') or [])}")

	st.subheader("Question Answering")
	question = st.text_input("Enter your question about the video")
	ask_btn = st.button("Ask")

	if ask_btn and question:
		retriever = st.session_state["retriever"]
		qa_prompt = load_qa_promt()

		# Build RetrievalQA lazily to avoid import-time issues
		from langchain.chains import RetrievalQA

		try:
			qa_chain = RetrievalQA.from_llm(load_llm() if load_llm else None, retriever=retriever, prompt=qa_prompt, return_source_documents=True)
		except Exception:
			# fallback: create RetrievalQA without return_source_documents
			qa_chain = RetrievalQA.from_llm(load_llm() if load_llm else None, retriever=retriever, prompt=qa_prompt)

		with st.spinner("Running QA..."):
			try:
				res = qa_chain({"query": question})
				answer = res.get("result") or res.get("output_text") or res.get("answer") or res
				st.write("### Answer")
				st.write(answer)
				if isinstance(res, dict) and "source_documents" in res:
					st.write("### Source documents (top-k)")
					for i, sd in enumerate(res["source_documents"]):
						st.write(f"--- doc {i} ---")
						st.write(sd.page_content[:1000])
			except Exception:
				# fallback to manual flow
				docs_for_q = retriever.get_relevant_documents(question)
				context = "\n\n---\n\n".join(d.page_content for d in docs_for_q)
				filled = qa_prompt.format(context=context, question=question)
				try:
					llm = load_llm() if load_llm else None
					if llm is not None:
							try:
								out = llm.invoke(filled)
							except Exception as e_invoke:
								out = f"LLM invocation failed: {e_invoke}"
					else:
						out = "LLM not configured. Please provide COHERE_API_KEY and implement load_llm."
				except Exception as e:
					out = f"Failed to run LLM: {e}"
				st.write("### Answer")
				st.write(out)

	st.subheader("Summarize")
	summarize_scope = st.radio("Summarize scope", options=["Full transcript", "Top-k relevant to a prompt"], index=0)
	summary_btn = st.button("Generate summary")

	if summary_btn:
		with st.spinner("Generating summary..."):
			if summarize_scope == "Full transcript":
				transcript_text = st.session_state["plain_text"]
			else:
				prompt_for_retrieval = st.text_input("Mini-prompt to retrieve top-k for summary", value="summarize main points")
				retriever = st.session_state["retriever"]
				top_docs = retriever.get_relevant_documents(prompt_for_retrieval)
				transcript_text = "\n\n".join(d.page_content for d in top_docs)

			# Use a simple summarization prompt
			from langchain.prompts import PromptTemplate

			SUMMARY_TEMPLATE = """
			Please summarize the following transcript in one concise paragraph:
			{transcript}
			"""
			summary_prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["transcript"])
			filled_summary = summary_prompt.format(transcript=transcript_text)

			try:
				llm = load_llm() if load_llm else None
				if llm is not None:
					try:
						summary = llm.invoke(filled_summary)
					except Exception as e_invoke:
						summary = f"LLM invocation failed: {e_invoke}"
				else:
					summary = "LLM not configured. Please provide COHERE_API_KEY and implement load_llm."
			except Exception as e:
				summary = f"Failed to generate summary: {e}"

			st.write("### Summary")
			st.write(summary)

else:
	st.info("Load a YouTube transcript to enable QA and summarization.")





