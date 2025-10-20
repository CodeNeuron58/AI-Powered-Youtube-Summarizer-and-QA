from langchain_cohere import ChatCohere
import os

from dotenv import load_dotenv
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY not found in .env file")


def load_llm():
    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-a-03-2025",  # model selection can be changed
    )

    return llm



