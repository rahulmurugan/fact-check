import json
import os
import re
import fitz  # PyMuPDF






def load_marketing_claims(filepath):
    """
    Loads and returns the marketing claims from a JSON file.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data.get("claims", [])


def load_clinical_docs(directory):
    """
    Loads clinical documents from the given directory.
    For PDFs, returns a dictionary where the key is the filename and the value is the file path.
    """
    docs = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            docs[filename] = os.path.join(directory, filename)
    return docs


def clean_text(text):
    """
    Performs basic text cleaning: stripping whitespace and normalizing spaces.
    """
    return ' '.join(text.strip().split())


def extract_pdf_text(filepath):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    If fitz is not available, raises an ImportError.
    """
    
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean_pdf_text(text):
    """
    Performs advanced cleaning on text extracted from PDFs.
    
    This function removes or normalizes:
      - Page numbers and headers/footers (common in clinical and regulatory PDFs)
      - Multiple consecutive newlines
      - URLs that may not add relevant content for matching
      - Non-text artifacts (like extra spaces, tab characters)
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove repeated headers/footers based on common patterns
    # (This might require fine-tuning depending on the actual header/footer styles)
    text = re.sub(r'Page \d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    
    # Remove any bracketed page numbers or section markers (e.g., [1], (1))
    text = re.sub(r'[\[\(]\s*\d+\s*[\]\)]', '', text)
    
    # Remove multiple newlines and replace with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Normalize whitespace
    text = clean_text(text)
    
    return text


def process_clinical_doc(filepath):
    """
    Extracts and cleans text from a clinical document PDF.
    
    Returns the cleaned text that is ready for analysis.
    """
    raw_text = extract_pdf_text(filepath)
    cleaned = clean_pdf_text(raw_text)
    return cleaned
