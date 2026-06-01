# llm-cost-optimizer

A real estate RAG chatbot that demonstrates 7 production token optimization techniques with a live cost dashboard. Built to show how a naive LLM pipeline burning ~$3,700/day can be cut by 70%+ through targeted engineering — not model downgrades.

---

## What it demonstrates

| Technique | Where | Impact |
|---|---|---|
| Prompt routing (regex/dict) | `router.py` | Amenity expansion at zero LLM cost |
| Reduced embedding dimensions | `embedder.py` | 768 dims instead of 1024 — cheaper, same quality |
| Client singleton | `embedder.py`, `pipeline.py` | One client init per process, not per request |
| JSON payload trimming | `pipeline.py` | 20 fields → 5 fields sent to LLM |
| Prefix caching | `pipeline.py` | Static system prompt ≥1024 tokens — 50% input cost discount on repeat calls |
| Re-ranker hybrid | `reranker.py` | Cross-encoder pre-filters 15 candidates; only ambiguous ones reach the LLM |
| Chat history compaction | `compactor.py` | Summarize history every 5 turns — context window stays flat |

Every technique is visible in real time: the sidebar shows baseline vs actual cost, and the terminal logs each pipeline step as it fires.

---

## Architecture

```
User query
    │
    ▼
router.py         regex + dict keyword expansion (free)
    │
    ▼
embedder.py       768-dim OpenAI embeddings + cosine search over 50 listings
    │
    ▼
reranker.py       cross-encoder scores all candidates
                  score > 0.7  → accepted (skip LLM judge)
                  0.15–0.7    → ambiguous (send to LLM)
                  score < 0.15 → discarded
    │
    ▼
pipeline.py       trimmed JSON + ≥1024-token system prompt + model routing
    │
    ▼
compactor.py      summarize history every 5 turns
    │
    ▼
app.py            Streamlit chat UI + live cost dashboard
```

---

## Stack

- **Python 3.11**
- **OpenAI API** — `text-embedding-3-small` (768 dims) + `gpt-4o-mini` / `gpt-4o`
- **sentence-transformers** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, no hosting)
- **numpy** — cosine similarity search (no external vector DB)
- **tiktoken** — token counting for before/after comparison
- **Streamlit** — UI + cost dashboard

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/llm-cost-optimizer
cd llm-cost-optimizer
pip install -r requirements.txt
```

Open `config.py` and add your OpenAI API key:

```python
OPENAI_API_KEY = "sk-..."
```

Then run:

```bash
streamlit run app.py
```

First load takes ~15 seconds — embeddings for 50 listings are computed at startup and the cross-encoder model is downloaded once and cached locally.

---

## Demo scenarios

Run these queries in order to hit every optimization.

### 1. `ocean view penthouse Miami`
**What fires:** prompt routing (2 rules: `ocean_view` + `penthouse`), re-ranker splits candidates, gpt-4o-mini selected.

Terminal output:
```
[ROUTER  ] Rules fired: ocean_view, penthouse → query expanded
[RERANK  ] Accepted: 4 | Ambiguous: 3 | Discarded: 8
[MODEL   ] Routing → gpt-4o-mini (simple query)
[COST    ] Actual: $0.00027 | Baseline: $0.00089 | Saved: 70%
```

### 2. `ocean view penthouse Miami` *(repeat)*
**What fires:** prefix cache hits — cached tokens jump from 0 to ~900. Same query, 50% cheaper input.

Terminal output:
```
[LLM     ] Input: 1166 tokens | Cached: 912 | Output: 143
[COST    ] Actual: $0.00019 | Baseline: $0.00089 | Saved: 79%
```

### 3. `3 bedroom pool Austin under 700k`
**What fires:** pool routing rule, re-ranker discards non-Austin/non-pool listings hard, JSON trimming clearly visible (3414 → 1461 tokens).

Terminal output:
```
[ROUTER  ] Rules fired: pool → query expanded
[TRIM    ] Full JSON: 3414 tokens → Trimmed: 1461 tokens
[RERANK  ] Accepted: 3 | Ambiguous: 2 | Discarded: 10
```

### 4. `compare investment roi Miami vs Austin`
**What fires:** model routing escalates to gpt-4o — `compare`, `roi`, `vs` trigger the complexity check.

Terminal output:
```
[ROUTER  ] No rules fired — query passed through
[MODEL   ] Routing → gpt-4o (complex query)
```

### 5. `show me something affordable in Boston` *(turn 5)*
**What fires:** history compaction — all previous turns summarized into one message.

Terminal output:
```
[COMPACT ] History summarized — 847 tokens saved
```

---

## Cost dashboard

The sidebar shows per-session:

- **Baseline cost** — what every query would cost with gpt-4o, full JSON, no caching, no re-ranker
- **Actual cost** — what it actually cost
- **You saved** — delta in dollars and percentage (displayed in green)
- Cached tokens, listings filtered, routing saves, compaction count

Each chat response also has a collapsible **Cost details** panel with per-call breakdown.

---

## Project structure

```
llm-cost-optimizer/
├── app.py              # Streamlit UI + cost dashboard
├── pipeline.py         # Orchestration + logging
├── router.py           # Regex/dict query expansion
├── embedder.py         # OpenAI embeddings + cosine search
├── reranker.py         # Cross-encoder re-ranking
├── compactor.py        # Chat history summarization
├── config.py           # API key + model/threshold settings
├── data/
│   └── listings.json   # 50 synthetic real estate listings
└── requirements.txt
```

---

## Configuration

All thresholds and model names are in `config.py`:

```python
EMBED_MODEL   = "text-embedding-3-small"
EMBED_DIMS    = 768           # reduced from 1024
FAST_MODEL    = "gpt-4o-mini"
SMART_MODEL   = "gpt-4o"
RERANKER_LOW  = 0.15          # discard threshold (sigmoid)
RERANKER_HIGH = 0.70          # accept threshold (sigmoid)
HISTORY_LIMIT = 5             # turns before compaction
```
