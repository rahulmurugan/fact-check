from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_matrix(clinical_texts):
    """
    Given a list of clinical texts, builds a TF-IDF matrix and returns both the matrix and the vectorizer.
    
    Parameters:
        clinical_texts (list of str): Each element is a full text from a clinical document.
    
    Returns:
        tfidf_matrix (scipy.sparse matrix): TF-IDF features for each clinical text.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    """
    # remove English stop words to improve the quality of our vectorization.
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(clinical_texts)
    return tfidf_matrix, vectorizer


def match_claim_to_docs(claim, clinical_texts, tfidf_matrix, vectorizer, top_k=3):
    """
    Matches a given claim text with clinical documents based on cosine similarity on their TF-IDF representations.
    
    Parameters:
        claim (str): The marketing claim text.
        clinical_texts (list of tuple): List of tuples in the format (document_name, document_text).
        tfidf_matrix (scipy.sparse matrix): Precomputed TF-IDF matrix for clinical document texts.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        top_k (int): Number of top matching documents to return.
    
    Returns:
        List of dictionaries containing:
            - document_name: the file name or ID of the clinical document.
            - matching_text: a snippet/summary (here we use the first 200 characters of the document text).
            - score: the cosine similarity score.
    """
    # Transform the claim text into the TF-IDF vector space.
    claim_vec = vectorizer.transform([claim])
    # Compute cosine similarity between the claim and all clinical documents.
    similarities = cosine_similarity(claim_vec, tfidf_matrix).flatten()
    # Retrieve the indices of the top_k scores (highest similarity values).
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    matches = []
    for idx in top_indices:
        doc_name, doc_text = clinical_texts[idx]
        # Pick a snippet (first 200 characters) from the document's cleaned text for reference.
        snippet = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
        match = {
            "document_name": doc_name,
            "matching_text": snippet,
            "score": float(similarities[idx])
        }
        matches.append(match)
    
    return matches


def build_clinical_texts_dict(clinical_docs, process_func):
    """
    Given a dictionary of clinical document file paths (from load_clinical_docs),
    this function processes each document using the provided processing function
    (such as process_clinical_doc from preprocess.py) and returns a list of tuples:
    (document_name, processed_text).
    
    Parameters:
        clinical_docs (dict): Dictionary where keys are document names and values are paths.
        process_func (function): Function that accepts a file path and returns cleaned text.
    
    Returns:
        List of tuples: [(doc_name, text), ...]
    """
    clinical_texts = []
    for doc_name, path in clinical_docs.items():
        try:
            text = process_func(path)
            clinical_texts.append((doc_name, text))
        except Exception as e:
            print(f"Error processing {doc_name}: {e}")
    return clinical_texts