import streamlit as st
import openai
import numpy as np
import json
import re
from typing import List, Dict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML/AI Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Knowledge base ────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {
        "id": 0,
        "topic": "Transformer Architecture",
        "content": """The Transformer architecture, introduced in 'Attention Is All You Need' (Vaswani et al., 2017),
revolutionised NLP by replacing recurrence with self-attention mechanisms. The core idea is that every token
in a sequence can attend to every other token in parallel, enabling the model to capture long-range dependencies
efficiently. The architecture consists of an encoder and decoder, each made of stacked layers containing
multi-head self-attention and feed-forward networks. Multi-head attention allows the model to jointly attend
to information from different representation subspaces at different positions. Positional encodings are added
to token embeddings to inject sequence order information, since the architecture has no inherent notion of order.
Layer normalisation and residual connections are used throughout to stabilise training. The Transformer became
the foundation for BERT, GPT, T5, and virtually every modern large language model."""
    },
    {
        "id": 1,
        "topic": "Retrieval-Augmented Generation (RAG)",
        "content": """Retrieval-Augmented Generation (RAG), introduced by Lewis et al. (2020), combines parametric
knowledge stored in model weights with non-parametric knowledge retrieved from an external corpus at inference
time. The pipeline has two phases: retrieval and generation. During retrieval, a dense vector search finds
the top-k most relevant documents for a query using embedding similarity (typically cosine similarity).
During generation, those documents are prepended to the prompt as context, grounding the model's response.
RAG solves two key problems: knowledge cutoffs (the model can access fresh documents) and hallucination
(the model is anchored to retrieved facts). Hybrid RAG combines dense retrieval (vector search) with sparse
retrieval (BM25 keyword search) and uses Reciprocal Rank Fusion to merge results. Cross-encoder reranking
further refines the top candidates. Production RAG systems measure quality with RAGAS metrics: faithfulness,
answer relevancy, and context recall."""
    },
    {
        "id": 2,
        "topic": "Large Language Models (LLMs)",
        "content": """Large Language Models are neural networks trained on massive text corpora using the
self-supervised objective of next-token prediction. GPT-3 (175B parameters) demonstrated that scale alone
could produce emergent capabilities including few-shot learning. Modern LLMs like GPT-4, Claude, Gemini,
and Llama are trained in stages: pre-training on internet-scale text, supervised fine-tuning (SFT) on
instruction-following examples, and RLHF (Reinforcement Learning from Human Feedback) to align outputs
with human preferences. Key concepts include: context window (maximum tokens the model can process),
temperature (controls output randomness, 0=deterministic), and tokenisation (text is split into subword
units using BPE or SentencePiece). Inference efficiency techniques include KV-caching, flash attention,
quantisation (INT8/INT4), and speculative decoding. Emergent abilities appear unpredictably at scale —
abilities not present in smaller models that suddenly appear in larger ones."""
    },
    {
        "id": 3,
        "topic": "Fine-Tuning & LoRA",
        "content": """Fine-tuning adapts a pre-trained model to a specific task by continuing training on
a smaller, task-specific dataset. Full fine-tuning updates all model weights and is computationally expensive.
Parameter-Efficient Fine-Tuning (PEFT) methods update only a small fraction of parameters. LoRA (Low-Rank
Adaptation) is the most widely used PEFT technique: it freezes pre-trained weights and injects trainable
low-rank decomposition matrices (A and B, where rank r << d) into each transformer layer. Only these small
matrices are trained, reducing trainable parameters by 10,000x while achieving comparable performance to
full fine-tuning. QLoRA extends LoRA with 4-bit quantisation of the base model, enabling fine-tuning of
70B parameter models on a single GPU. DPO (Direct Preference Optimisation) is an alternative to RLHF that
fine-tunes models on preference pairs (chosen vs rejected responses) without a separate reward model.
The Hugging Face PEFT library and TRL library are the standard tools for LoRA and DPO training."""
    },
    {
        "id": 4,
        "topic": "Embeddings & Vector Search",
        "content": """Embeddings are dense vector representations of text that capture semantic meaning.
Similar meanings cluster together in vector space — 'king' and 'queen' are closer to each other than
to 'car'. Text embedding models like OpenAI's text-embedding-3-small, Sentence-BERT, and E5 encode
sentences into fixed-size vectors (typically 384–1536 dimensions). Cosine similarity measures the angle
between two vectors, returning 1 for identical meaning and 0 for orthogonal. Vector databases (FAISS,
Pinecone, Weaviate, Chroma) enable approximate nearest-neighbour (ANN) search at scale using algorithms
like HNSW (Hierarchical Navigable Small World graphs) and IVF (Inverted File Index). FAISS (Facebook AI
Similarity Search) is an open-source library optimised for billion-scale similarity search. The key
performance tradeoff is recall vs latency: HNSW is fast but approximate; flat exhaustive search is exact
but slow. In production RAG, embeddings power the dense retrieval leg of hybrid search."""
    },
    {
        "id": 5,
        "topic": "Attention Mechanism",
        "content": """Attention allows a model to focus on relevant parts of the input when producing
each output token. Scaled dot-product attention computes: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V,
where Q (queries), K (keys), and V (values) are linear projections of the input. The sqrt(d_k) scaling
prevents vanishing gradients in high-dimensional spaces. Multi-head attention runs h parallel attention
heads, each learning different relational patterns, then concatenates and projects the results.
Self-attention means Q, K, V all come from the same sequence — each token attends to all others.
Cross-attention (used in encoder-decoder models) lets the decoder attend to encoder outputs.
Causal (masked) self-attention prevents tokens from attending to future positions, essential for
autoregressive generation. Flash Attention (Dao et al., 2022) is an IO-aware exact attention algorithm
that reduces memory from O(N²) to O(N) by tiling computations, enabling training on much longer sequences."""
    },
    {
        "id": 6,
        "topic": "Evaluation Metrics for LLMs",
        "content": """Evaluating LLMs is notoriously hard because natural language has no single correct answer.
Automatic metrics include: BLEU (n-gram overlap, originally for translation), ROUGE (recall-oriented,
used for summarisation), BERTScore (semantic similarity via embeddings), and perplexity (how surprised
the model is by test text). RAGAS is a framework specifically for RAG evaluation with three metrics:
Faithfulness (does the answer use only retrieved context, catching hallucinations?), Answer Relevancy
(does the answer address the question?), and Context Recall (did retrieval find the needed chunks?).
LLM-as-judge uses a strong model (GPT-4) to score outputs on rubrics — increasingly standard for
open-ended generation. Human evaluation remains the gold standard but is expensive and slow.
MT-Bench and MMLU are popular benchmarks for general capability. Evals should always include adversarial
cases: out-of-scope questions, ambiguous queries, and multi-hop reasoning."""
    },
    {
        "id": 7,
        "topic": "Prompt Engineering",
        "content": """Prompt engineering is the practice of designing inputs to LLMs to elicit desired outputs.
Zero-shot prompting gives the model a task with no examples. Few-shot prompting includes 2–5 examples in
the prompt, dramatically improving performance on structured tasks. Chain-of-thought (CoT) prompting,
introduced by Wei et al. (2022), instructs the model to reason step by step before answering, improving
performance on arithmetic and logical tasks. The phrase 'Let's think step by step' is a zero-shot CoT trigger.
Structured output prompting uses JSON schemas or XML tags to enforce output format. System prompts set
model behaviour and persona. Temperature controls randomness: 0 for deterministic factual tasks, 0.7–1.0
for creative tasks. Top-p (nucleus sampling) samples from the smallest set of tokens whose cumulative
probability exceeds p. Key techniques: role assignment ('You are an expert in...'), negative constraints
('Do not include...'), and output format specification ('Respond only in JSON')."""
    },
    {
        "id": 8,
        "topic": "Reinforcement Learning from Human Feedback (RLHF)",
        "content": """RLHF aligns language models with human preferences through three stages. First, supervised
fine-tuning (SFT): the base model is fine-tuned on high-quality human-written demonstrations. Second,
reward model training: human labellers rank model outputs by preference; a separate reward model is trained
to predict these preferences. Third, RL optimisation: the SFT model is further trained using PPO (Proximal
Policy Optimisation) to maximise reward model scores, with a KL-divergence penalty to prevent the model
drifting too far from the SFT baseline. InstructGPT (Ouyang et al., 2022) demonstrated RLHF at scale,
showing that a 1.3B RLHF model was preferred over a 175B base model. Constitutional AI (Anthropic) extends
RLHF with an AI-generated critique-revision loop guided by a set of principles. DPO simplifies RLHF by
directly optimising on preference pairs without a separate reward model, using a closed-form solution
derived from the Bradley-Terry preference model."""
    },
    {
        "id": 9,
        "topic": "Model Quantisation & Inference Optimisation",
        "content": """Quantisation reduces model size and speeds up inference by representing weights in lower
precision. FP32 (32-bit float) → FP16/BF16 (16-bit) halves memory with minimal quality loss.
INT8 quantisation (8-bit integers) further halves memory; libraries like bitsandbytes enable this.
INT4/NF4 quantisation (4-bit) is used in QLoRA, compressing a 7B model to ~3.5GB. Post-training
quantisation (PTQ) quantises a trained model without retraining. Quantisation-aware training (QAT)
simulates low precision during training for better accuracy. Beyond quantisation: KV-cache stores
previously computed key-value pairs to avoid recomputation during autoregressive decoding.
Speculative decoding uses a small draft model to propose tokens that a larger model verifies in parallel,
achieving 2–3x speedup. Flash Attention reduces memory bandwidth usage. vLLM uses PagedAttention for
efficient KV-cache management, enabling high-throughput serving. GGUF format (used by llama.cpp)
enables quantised inference on consumer CPUs."""
    },
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0f1117; }
  .stApp { background: #0f1117; }

  .hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2d3561;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
  }
  .hero h1 { font-size: 2rem; font-weight: 700; color: #e2e8f0; margin: 0; }
  .hero p  { color: #94a3b8; margin: 0.5rem 0 0; font-size: 1rem; }

  .metric-row {
    display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap;
  }
  .metric-card {
    flex: 1; min-width: 100px;
    background: #1e2433;
    border: 1px solid #2d3561;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #60a5fa; }
  .metric-card .lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }

  .chunk-card {
    background: #1a1f2e;
    border: 1px solid #2d3561;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.6;
  }
  .chunk-title {
    font-weight: 600; color: #60a5fa; font-size: 0.8rem;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
  }
  .score-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-bottom: 0.5rem;
  }

  .answer-box {
    background: #1a1f2e;
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.75;
    margin-bottom: 1rem;
  }

  .tag {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 3px;
  }

  .pipeline-step {
    display: inline-block;
    background: #1e2433;
    border: 1px solid #2d3561;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 2px;
  }
  .pipeline-arrow { color: #3b82f6; margin: 0 2px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── OpenAI client setup ───────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


# ── Embedding & retrieval ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_embeddings(_client, texts: List[str]) -> np.ndarray:
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([r.embedding for r in response.data])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def bm25_scores(query: str, docs: List[str], k1=1.5, b=0.75) -> np.ndarray:
    query_terms = query.lower().split()
    tokenised = [d.lower().split() for d in docs]
    avg_dl = np.mean([len(d) for d in tokenised])
    scores = np.zeros(len(docs))
    for term in query_terms:
        df = sum(1 for d in tokenised if term in d)
        if df == 0:
            continue
        idf = np.log((len(docs) - df + 0.5) / (df + 0.5) + 1)
        for i, doc in enumerate(tokenised):
            tf = doc.count(term)
            dl = len(doc)
            scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return scores


def hybrid_retrieve(query: str, client, top_k: int = 3) -> List[Dict]:
    texts = [c["content"] for c in KNOWLEDGE_BASE]
    doc_embeddings = get_embeddings(client, texts)
    query_embedding = get_embeddings(client, [query])[0]

    vec_scores = cosine_similarity(query_embedding, doc_embeddings)
    kw_scores = bm25_scores(query, texts)

    # Reciprocal Rank Fusion
    vec_ranks = np.argsort(-vec_scores)
    kw_ranks = np.argsort(-kw_scores)
    rrf = np.zeros(len(texts))
    for rank, idx in enumerate(vec_ranks):
        rrf[idx] += 1 / (60 + rank + 1)
    for rank, idx in enumerate(kw_ranks):
        rrf[idx] += 1 / (60 + rank + 1)

    top_indices = np.argsort(-rrf)[:top_k]
    results = []
    for idx in top_indices:
        chunk = KNOWLEDGE_BASE[idx].copy()
        chunk["rrf_score"] = round(float(rrf[idx]), 4)
        chunk["vec_score"] = round(float(vec_scores[idx]), 3)
        results.append(chunk)
    return results


def generate_answer(query: str, chunks: List[Dict], client) -> str:
    context = "\n\n---\n\n".join([
        f"[SOURCE:{c['id']} | {c['topic']}]\n{c['content']}"
        for c in chunks
    ])
    prompt = f"""You are an expert AI/ML educator. Answer the question using ONLY the provided context.
For every factual claim, cite the source using [SOURCE:id]. Be concise but complete.
If the answer isn't in the context, say "This topic isn't covered in my current knowledge base."

Context:
{context}

Question: {query}

Answer (with citations):"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content


def highlight_citations(text: str, chunks: List[Dict]) -> str:
    topic_map = {str(c["id"]): c["topic"] for c in chunks}
    def replace(m):
        cid = m.group(1)
        topic = topic_map.get(cid, f"chunk {cid}")
        return f"**[{topic}]**"
    return re.sub(r'\[SOURCE:(\d+)\]', replace, text)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 1, 5, 3)
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_pipeline = st.toggle("Show pipeline trace", value=True)

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")
    for chunk in KNOWLEDGE_BASE:
        st.markdown(f"<span class='tag'>{chunk['topic']}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    **Retrieval:** Hybrid BM25 + vector search with Reciprocal Rank Fusion  
    **Embeddings:** OpenAI text-embedding-3-small  
    **Generation:** GPT-4o-mini with citation enforcement  
    **Stack:** Python · Streamlit · OpenAI API · NumPy
    """)
    st.markdown("---")
    st.caption("Built by [Your Name] · [GitHub](https://github.com) · [LinkedIn](https://linkedin.com)")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 AI/ML Knowledge Base</h1>
  <p>Ask anything about Machine Learning, LLMs, RAG, Fine-tuning, Transformers & more</p>
</div>
""", unsafe_allow_html=True)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="val">10</div><div class="lbl">Topics indexed</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="val">Hybrid</div><div class="lbl">Retrieval method</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="val">RRF</div><div class="lbl">Fusion algorithm</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="val">GPT-4o</div><div class="lbl">Generation model</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Suggested questions
st.markdown("**💡 Try asking:**")
suggestions = [
    "How does the Transformer architecture work?",
    "What is RAG and how does hybrid retrieval improve it?",
    "Explain LoRA and how it reduces training cost",
    "What is RLHF and how does DPO differ from it?",
    "How does quantisation speed up LLM inference?",
]
cols = st.columns(len(suggestions))
for i, (col, q) in enumerate(zip(cols, suggestions)):
    with col:
        if st.button(q, key=f"sug_{i}", use_container_width=True):
            st.session_state["query"] = q

# Query input
query = st.text_input(
    "Ask a question",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What is the difference between LoRA and full fine-tuning?",
    label_visibility="collapsed",
)

if query:
    client = get_client()
    if not client:
        st.error("⚠️ OpenAI API key not found. Add it to Streamlit secrets as OPENAI_API_KEY.")
        st.stop()

    with st.spinner("Retrieving and generating..."):
        chunks = hybrid_retrieve(query, client, top_k=top_k)
        answer = generate_answer(query, chunks, client)

    # Pipeline trace
    if show_pipeline:
        st.markdown("**🔍 Pipeline trace**")
        st.markdown(f"""
        <span class='pipeline-step'>Query</span>
        <span class='pipeline-arrow'>→</span>
        <span class='pipeline-step'>BM25 sparse search</span>
        <span class='pipeline-arrow'>+</span>
        <span class='pipeline-step'>Vector dense search</span>
        <span class='pipeline-arrow'>→</span>
        <span class='pipeline-step'>RRF fusion</span>
        <span class='pipeline-arrow'>→</span>
        <span class='pipeline-step'>Top {top_k} chunks</span>
        <span class='pipeline-arrow'>→</span>
        <span class='pipeline-step'>GPT-4o-mini</span>
        <span class='pipeline-arrow'>→</span>
        <span class='pipeline-step'>Answer + citations</span>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Answer
    st.markdown("**🤖 Answer**")
    highlighted = highlight_citations(answer, chunks)
    st.markdown(f'<div class="answer-box">{highlighted}</div>', unsafe_allow_html=True)

    # Retrieved chunks
    if show_chunks:
        st.markdown(f"**📄 Retrieved chunks** (top {top_k} by RRF score)")
        for chunk in chunks:
            st.markdown(f"""
            <div class="chunk-card">
              <div class="chunk-title">{chunk['topic']}</div>
              <span class='score-pill'>RRF: {chunk['rrf_score']} · Cosine: {chunk['vec_score']}</span>
              <div>{chunk['content'][:280]}...</div>
            </div>
            """, unsafe_allow_html=True)
