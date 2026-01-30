# 🎬 Video-RAG-Analyst

This application allows users to **summarize** and **ask questions** about YouTube videos. It uses Retrieval-Augmented Generation (RAG) to fetch transcripts, store them in a vector database, and generate answers or summaries using the **Cohere** Large Language Model (LLM).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vidbrief-ai.streamlit.app/) &nbsp;

**Live Demo:** [vidbrief-ai.streamlit.app](https://vidbrief-ai.streamlit.app/)

## ✨ Features

-   **📽️ Transcript Extraction**: Automatically fetches English transcripts from YouTube videos.
-   **🔍 RAG-based QA**: Ask any question about the video content and get answers with source citations.
-   **📝 Summarization**: Generate summaries for the entire video or specific topics.
-   **⚡ Streamlit UI**: A user-friendly, interactive web interface.


> **Note**: Currently supports videos with **English subtitles** only.

## 🛠️ Architecture

-   **Frontend**: Streamlit
-   **LLM**: Cohere (via `langchain-cohere`)
-   **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
-   **Vector Store**: FAISS
-   **Framework**: LangChain

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/CodeNeuron58/AI-Powered-Youtube-Summarizer-and-QA.git

    cd AI-Powered-Youtube-Summarizer-and-QA
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv

    # Windows
    .\venv\Scripts\activate

    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration

You need a **Cohere API Key** to use the LLM features.

1.  Create a `.env` file in the root directory:
    ```bash
    touch .env  # or create manually
    ```
2.  Add your API key:
    ```env
    COHERE_API_KEY=your_cohere_api_key_here
    ```

*Alternatively, you can enter the API Key directly in the Streamlit Sidebar.*

## 🏃 Usage

### Streamlit UI (Recommended)
Launch the interactive web app:
```bash
streamlit run streamlit_ui.py
```


## 📁 Project Structure

```
├── src/                # Core Logic
│   ├── chains/         # LLM configuration (Cohere)
│   ├── preprocessing/  # Text splitting & Embeddings
│   ├── promt/          # Prompt Templates
│   ├── retriever/      # Vector Store & Retrieval
│   └── transcript/     # YouTube Transcript Fetching
├── streamlit_ui.py     # Main Streamlit Application
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```