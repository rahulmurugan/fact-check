import faiss
import numpy as np

def build_faiss_index(embeddings: list) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from a list of embedding vectors.
    
    Parameters:
        embeddings (list): A list of embeddings (each embedding is a list of floats).
        
    Returns:
        faiss.IndexFlatL2: FAISS index built on the embeddings using L2 distance.
    """
    # Convert embeddings list into a numpy array with datatype float32.
    emb_array = np.array(embeddings).astype("float32")
    # Determine the vector dimension from the first element.
    dimension = emb_array.shape[1]
    # Create an index with L2 (Euclidean) distance.
    index = faiss.IndexFlatL2(dimension)
    # Add the embeddings into the index.
    index.add(emb_array)
    return index

def query_faiss_index(index: faiss.IndexFlatL2, query_embedding: list, top_k: int = 3):
    """
    Queries the FAISS index for the top-k most similar embeddings.
    
    Parameters:
        index (faiss.IndexFlatL2): The FAISS index built from embeddings.
        query_embedding (list): The query embedding as a list of floats.
        top_k (int): The number of nearest neighbors to retrieve.
        
    Returns:
        List[Tuple[int, float]]: A list of tuples where each tuple is (index in the database, distance).
    """
    # Prepare the query vector by converting it to a numpy array of type float32.
    query_np = np.array([query_embedding]).astype("float32")
    # Perform the search on the FAISS index.
    distances, indices = index.search(query_np, top_k)
    # Combine the index and distance results into a list of tuples.
    return list(zip(indices[0], distances[0]))


