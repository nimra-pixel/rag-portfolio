# 🧠 AI/ML Knowledge Base — RAG Application

A production-grade Retrieval-Augmented Generation (RAG) app built as part of an AI Engineer portfolio. Ask natural language questions about Machine Learning, LLMs, Transformers, Fine-tuning, and more.

🚀 **[Live Demo →](https://your-app.streamlit.app)**

---

## Features

- **Hybrid retrieval** — BM25 sparse search + OpenAI vector search combined with Reciprocal Rank Fusion
- **Citation enforcement** — every answer cites the source chunks used
- **Pipeline tracing** — visualises each step of the retrieval and generation process
- **10 ML/AI topics** indexed: Transformers, RAG, LLMs, LoRA, Attention, Embeddings, RLHF, Quantisation, Prompt Engineering, Evals

## Architecture

```
Query → BM25 sparse search ┐
                            ├→ RRF fusion → Top-k chunks → GPT-4o-mini → Answer + citations
Query → Vector dense search ┘
```

| Component | Technology |
|-----------|-----------|
| Embeddings | OpenAI text-embedding-3-small |
| Sparse retrieval | BM25 (Okapi BM25, pure NumPy) |
| Dense retrieval | Cosine similarity over embedding vectors |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Generation | GPT-4o-mini, temperature=0.1 |
| Frontend | Streamlit |

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/rag-portfolio
cd rag-portfolio
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your key
mkdir -p .streamlit
echo '[secrets]\nOPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml

streamlit run app.py
```

## Deploy to Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `OPENAI_API_KEY` in the secrets manager
5. Deploy — get a public URL in 2 minutes

## Stack

`Python` `Streamlit` `OpenAI API` `NumPy`

---

Built by [Your Name](https://linkedin.com/in/yourprofile)
