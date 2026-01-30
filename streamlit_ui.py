import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project modules
from src.transcript.get_youtube_transcript import get_transcript
from src.preprocessing.transcipt_splitter import split_transcript
from src.retriever.retrieve_docs import retrieve_documents
from src.chains.LLM import load_llm
from src.preprocessing.embeddings import embeddings as load_embeddings
from src.promt.qa_promt import load_qa_promt
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Page Configuration
st.set_page_config(
    page_title="Video-RAG-Analyst",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED RESOURCES ---
# We use st.cache_resource for objects that should persist across reruns/sessions appropriately (like models)

@st.cache_resource(show_spinner="Loading Language Model...")
def get_cached_llm(api_key):
    """
    Cache the LLM loading to avoid re-initializing connection on every run.
    Note: We pass api_key to ensure it invalidates if key changes, 
    though strictly load_llm checks env. 
    """
    # Temporarily set env if passed explicitly (handling dynamic updates)
    if api_key:
        os.environ["COHERE_API_KEY"] = api_key
    return load_llm()

@st.cache_resource(show_spinner="Loading Embedding Model...")
def get_cached_embeddings():
    """Cache the heavy HuggingFace embedding model."""
    return load_embeddings()

# ------------------------

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    h1 {
        color: #FF0000;
        text-align: center;
    }
    .stButton>button {
        background-color: #FF0000;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("🎬 Video-RAG-Analyst")
st.markdown("Extract insights, ask questions, and summarize YouTube videos in seconds.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Handling
    api_key_env = os.getenv("COHERE_API_KEY")
    if not api_key_env:
        api_key_input = st.text_input("Enter Cohere API Key", type="password", help="Get your key at https://cohere.com/")
        if api_key_input:
            os.environ["COHERE_API_KEY"] = api_key_input
            api_key_env = api_key_input
            # Clear cache if key updates
            get_cached_llm.clear()
    
    if api_key_env:
        st.success("API Key loaded successfully!")
    else:
        st.warning("Please provide a Cohere API Key to use LLM features.")

    st.markdown("---")
    k_val = st.slider("Retrieval Depth (k)", min_value=1, max_value=10, value=5, help="Number of text chunks to retrieve for QA.")
    
    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    st.warning("Currently supports videos with **English subtitles** only.")
    st.info("This app uses Cohere for LLM generation and FAISS for vector retrieval.")

# Session State Initialization
if "transcript" not in st.session_state:
    st.session_state["transcript"] = None
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None
if "docs" not in st.session_state:
    st.session_state["docs"] = None

# Main Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Load Video")
    video_url = st.text_input("Paste YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Load Transcript (Process Video)"):
        if not video_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("Fetching transcript and building knowledge base..."):
                try:
                    # 1. Fetch Transcript
                    transcript_text = get_transcript(video_url)
                    st.session_state["transcript"] = transcript_text
                    
                    # 2. Split and Store
                    # Load cached embedding model first
                    emb_model = get_cached_embeddings()
                    
                    docs = split_transcript(transcript_text)
                    st.session_state["docs"] = docs
                    
                    # 3. Build Retriever (Pass cached embedding model)
                    vector_store, retriever = retrieve_documents(docs, k=k_val, embeddings_model=emb_model)
                    st.session_state["vector_store"] = vector_store
                    st.session_state["retriever"] = retriever
                    
                    st.success(f"Successfully loaded video! Processed {len(docs)} text chunks.")
                    
                except ValueError as ve:
                    st.error(f"{str(ve)}")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")

# Display Video if available
if video_url and "youtube.com" in video_url:
    st.video(video_url)

# Functionality Section (Only if loaded)
if st.session_state["retriever"]:
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🤖 Question Answering", "📝 Summarization"])
    
    with tab1:
        st.subheader("Ask functionality")
        question = st.text_input("Ask a question about the video:")
        
        if st.button("Get Answer"):
            if not question:
                st.warning("Please enter a question.")
            elif not api_key_env:
                st.error("Cohere API Key is missing. Please check the sidebar.")
            else:
                with st.spinner("Consulting the oracle..."):
                    try:
                        # Setup QA Chain
                        qa_prompt = load_qa_promt()
                        retriever = st.session_state["retriever"]
                        # Update retriever k if changed
                        retriever.search_kwargs["k"] = k_val
                        
                        # Load Cached LLM
                        llm = get_cached_llm(api_key_env)
                        
                        # Creating chain
                        if llm:
                            qa_chain = RetrievalQA.from_llm(
                                llm=llm,
                                retriever=retriever,
                                prompt=qa_prompt,
                                return_source_documents=True
                            )
                            
                            # Run Chain
                            res = qa_chain({"query": question})
                            answer = res.get("result") or res.get("output_text")
                            sources = res.get("source_documents", [])
                            
                            st.markdown("### Answer")
                            st.write(answer)
                            
                            with st.expander("View Source Documents"):
                                for i, doc in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content)
                                    st.markdown("---")
                        else:
                            st.error("Failed to load LLM.")
                                
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

    with tab2:
        st.subheader("Generate Summary")
        summary_type = st.radio("Choose Summary Type:", ["Entire Video", "Based on Search Query"])
        
        if summary_type == "Based on Search Query":
            search_query = st.text_input("Enter a topic or query to focus the summary on:")
        
        if st.button("Generate Summary"):
            if not api_key_env:
                st.error("Cohere API Key is missing.")
            else:
                with st.spinner("Summarizing..."):
                    try:
                        context_text = ""
                        if summary_type == "Entire Video":
                             context_text = st.session_state["transcript"]
                             # Truncate if insanely large to prevent API errors?
                             # Cohere has large context window usually, but let's be safe-ish or let it fail gracefully.
                        else:
                             if not search_query:
                                 st.warning("Please enter a search query.")
                                 st.stop()
                             retriever = st.session_state["retriever"]
                             relevant_docs = retriever.get_relevant_documents(search_query)
                             context_text = "\n\n".join([d.page_content for d in relevant_docs])

                        if context_text:
                            # Simple summarization prompt
                            SUMMARY_TEMPLATE = """
                            You are an expert summarizer. Please provide a comprehensive and structured summary of the following text.
                            
                            Text:
                            {text}
                            
                            Summary:
                            """
                            prompt = PromptTemplate(template=SUMMARY_TEMPLATE, input_variables=["text"])
                            # Truncate to avoid massive overload if needed, e.g. 50k chars
                            chain_input = prompt.format(text=context_text[:50000]) 
                            
                            # Load Cached LLM
                            llm = get_cached_llm(api_key_env)
                            
                            if llm:
                                summary = llm.invoke(chain_input).content
                                st.markdown("### Summary")
                                st.write(summary)
                            
                    except Exception as e:
                        st.error(f"Summary failed: {str(e)}")

else:
    with col2:
        st.info("👈 Please load a video URL to start.")
        st.write("### Instructions:")
        st.write("1. Get a YouTube URL.")
        st.write("2. Paste it in the input box.")
        st.write("3. Click 'Load Transcript'.")
        st.write("4. Ask questions or generate summaries in the tabs below.")
