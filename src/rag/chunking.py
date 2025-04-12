def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Splits input text into chunks of up to 'chunk_size' words, with an overlap of 'overlap' words.
    
    Parameters:
        text (str): The input text to be split.
        chunk_size (int): Maximum number of words per chunk.
        overlap (int): Number of words that overlap between consecutive chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    # If the text is shorter than the chunk size, return the text as-is
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # If we have reached or exceeded the word list, break
        if end >= len(words):
            break
        # Move start pointer for next chunk, with the specified overlap
        start = end - overlap

    return chunks