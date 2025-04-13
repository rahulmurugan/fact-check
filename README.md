# fact-check#  Fact Check Solution Guide

This project matches marketing claims from a flu vaccine product (Flublok) against supporting evidence from clinical literature. It explores three approaches to build a fact-checking system:

1. **Basic TF-IDF Matching**
2. **Semantic RAG (FAISS + DeepSeek-Qwen)**
3. **LLM-Inference via Ollama + FAISS Retrieval**

---
## 📁 Project Directory Structure

```bash
solstice-fact-check/
│
├── data/
│   └── Marketing File Page/              # Contains Flublok_Claims.json
│
├── outputs/                              # Output from Ollama-based inference
│   ├── ollama_results.json
│   └── ollama2_results.json
│
├── results/                              # Output from basic and RAG approaches
│   ├── basic_results.json
│   └── rag_results.json
│
├── preprocessed/                         # Extracted text, tables, and figures
│
├── vectordb/                             # FAISS index and serialized docstore
│   ├── index.faiss
│   └── docstore.pkl
│
├── src/                                  # Source code modules
│   ├── matcher.py                        # TF-IDF logic
│   ├── preprocess.py                     # PDF processing
│   ├── setup_path.py
│   ├── utils.py
│   └── rag/
│       ├── chunking.py
│       ├── embedding.py
│       ├── generation.py
│       ├── preprocess1.py
│       ├── retrieval.py
│
├── main1.ipynb                           # Basic TF-IDF notebook
├── main(RAG).ipynb                       # RAG pipeline using HuggingFace model
├── ollamaRAG.ipynb                       # Ollama-based LLM pipeline
│
├── README.md
├── requirements.txt
├── setup.py
│
├── Basic_approach.pdf
├── Exercise Description.pdf
├── RAG_method.pdf
└── Fact_check_ollama.pdf
```

## ⚙️ 1. Basic Approach (TF-IDF + Cosine Similarity)

This method builds a simple document retrieval pipeline using:

- **TF-IDF vectorization** of clinical texts and claims.
- **Cosine similarity** to rank matches.

Each claim is compared against all extracted clinical paragraphs. The top-3 matches are returned with similarity scores.

### ✅ Highlights

- Fast and interpretable
- Good baseline for comparison

### ⚠️ Limitations

- No semantic understanding of context or paraphrasing
- Lower quality matches (typically score 0.1–0.3)

📄 Output: `results/basic_results.json`

---

## 🧠 2. RAG Pipeline (FAISS + DeepSeek-Qwen LLM)

This approach combines semantic retrieval with local LLM-based generation.

### 🔧 Components

- **Retrieval**: Text is embedded using `MiniLM-L3-v2` and stored in a FAISS index.
- **LLM Inference**: Retrieved candidates are passed to:
  
  ```python
  model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  llm_pipeline = pipeline("text-generation", model=model_id, max_length=256, device=-1)

This HuggingFace pipeline runs fully locally and selects supporting snippets per claim.

### ✅ Highlights
- Open-source and local-first (no API keys required)
- Lightweight and CPU-friendly
- Clean JSON output with matched source texts

### ⚠️ Limitations
- Does not output `supports: true/false` or reasoning
- Relies heavily on good FAISS retrieval + prompt context

📄 **Output**: `results/rag_results.json`

---

## 🤖 3. Ollama + FAISS Retrieval (LLM-Inference)

This is the most advanced method, using:

- **SBERT + FAISS** for top-k semantic retrieval
- **Mistral LLM via Ollama** for deeper reasoning and claim evaluation

### 🔹 Version 1: `ollama_results.json`
- Returns only snippets the model considers as supporting the claim
- Efficient but lacks justification

### 🔸 Version 2: `ollama2_results.json`
Each snippet includes:
- `supports`: true/false
- `reason`: short explanation of decision

### ✅ Highlights
- Most explainable method
- Good for audits and regulatory submission
- Avoids fluency bias by using strict LLM logic

### ⚠️ Limitations
- Slower (multiple LLM calls per claim)
- High RAM usage (Mistral is heavy on CPU)

📄 **Output**: `outputs/ollama_results.json`, `outputs/ollama2_results.json`

---

## 📊 Method Comparison

| Method           | Semantic | LLM Used                          | Reasoning | Score Quality     | Output File             |
|------------------|----------|-----------------------------------|-----------|--------------------|--------------------------|
| TF-IDF           | ❌        | ❌                                 | ❌         | Low (0.1–0.3)       | `basic_results.json`     |
| RAG (DeepSeek)   | ✅        | DeepSeek-R1-Distill-Qwen (1.5B)   | ❌         | Medium (0.4–0.7)    | `rag_results.json`       |
| Ollama v1        | ✅        | Mistral via Ollama                | ❌         | High               | `ollama_results.json`    |
| Ollama v2        | ✅        | Mistral via Ollama                | ✅         | High               | `ollama2_results.json`   |

---

## 🚧 CoRAG and ToRAG (Future Work)

We are exploring multi-turn reasoning strategies to go beyond flat retrieval:

- **CoRAG**: Iterative QA chain that continues until confident
- **ToRAG**: Tree of Thought prompting with branch ranking and follow-up generation

### ⚠️ These methods require:
- Many LLM calls per claim
- Deep reasoning trees and dynamic branching

---

## 🚫 Current Limitation

Due to the memory load of multi-step Mistral inference, local execution causes:
- RAM exhaustion
- Kernel crashes
- Slow iteration and batching issues

---

## ☁️ Planned Deployment

We plan to move CoRAG/ToRAG to the cloud using:
- Google Colab (GPU)
- Hugging Face Inference Endpoints
- Replicate or Modal

This will support:
- Scalable, distributed LLM inference
- Deterministic, traceable outputs
- Production-ready experimentation

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
