from google import genai
import json
import re

def get_embedding(api_key: str, text: str, model: str = "gemini-2.0-flash") -> list:
    """
    Retrieves an embedding vector for the provided text using the Gemini model.
    This function calls the Gemini API via the google.genai client.
    
    Parameters:
        api_key (str): Your Gemini API key.
        text (str): The text to embed.
        model (str): Model name to be used for generating embeddings. Defaults to "gemini-2.0-flash".

    Returns:
        List[float]: The embedding vector as a list of numbers.
    
    Note:
        This implementation assumes that you can instruct the model to output the embedding
        in JSON format. If the response contains additional commentary text, the function
        extracts the JSON block using regular expressions.
    """
    client = genai.Client(api_key=api_key)
    prompt = f"Generate a numerical embedding vector in JSON format for the following text:\n\n{text}"
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    
    response_text = response.text.strip()
    
    # Try to extract a JSON block if it's wrapped in backticks or extra commentary.
    # This regex looks for a block starting with '```json' and ending with '```'.
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # If no code block markers are found, assume the entire response is JSON.
        json_str = response_text

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse embedding from response: {response_text}. Error: {e}")
    
    # Expecting the JSON to contain an "embedding" field (modify if needed).
    if "embedding" in parsed and isinstance(parsed["embedding"], list):
        embedding = parsed["embedding"]
    else:
        # If the JSON is directly a list, use it.
        if isinstance(parsed, list):
            embedding = parsed
        else:
            raise ValueError("Returned embedding is not in the expected format (not a list of numbers).")
    
    # Validate the embedding elements are numbers.
    if all(isinstance(x, (int, float)) for x in embedding):
        return embedding
    else:
        raise ValueError("Parsed embedding does not consist entirely of numbers.")
