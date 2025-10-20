# langchain hub
from langchain.prompts import PromptTemplate


def load_qa_promt():
    
    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    qa_promt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    return qa_promt
