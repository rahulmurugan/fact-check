
from google import genai

def generate_answer(api_key: str, model: str, query: str, retrieved_context: list, max_tokens: int = 150) -> str:
    """
    Generates an answer from Gemini using the provided query and retrieved context.
    
    Parameters:
        api_key (str): Your Gemini API key.
        model (str): The model name to use, e.g., "gemini-2.0-flash".
        query (str): The user query or claim.
        retrieved_context (list): A list of text snippets representing retrieved context.
        max_tokens (int): (Optional) Maximum number of tokens for the output answer.
        
    Returns:
        str: The generated answer text.
    
    Workflow:
        1. The function concatenates the retrieved context into a single string.
        2. It builds a prompt that instructs the model to generate a concise answer using the context.
        3. It sends the prompt to the Gemini API via the google.genai client.
        4. It returns the model’s textual output.
    """
    # Initialize the client using your API key.
    client = genai.Client(api_key=api_key)
    
    # Combine retrieved context into a single text block.
    context = "\n".join(retrieved_context)
    
    # Build a prompt that provides the context and then asks the question.
    prompt = (
        f"Using the following context, provide a concise and informative answer:\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\nAnswer:"
    )
    
    # Generate the answer using the provided model.
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    
    return response.text