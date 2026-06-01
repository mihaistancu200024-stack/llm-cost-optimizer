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
**What fires:** prompt routing (2 rules), re-ranker splits candidates, gpt-4o-mini selected.

Terminal output:
```
[ROUTER  ] Rules fired: ocean_view, penthouse → query expanded
[RERANK  ] Accepted: 4 | Ambiguous: 3 | Discarded: 8
[MODEL   ] Routing → gpt-4o-mini (simple query)
[COST    ] Actual: $0.00027 | Baseline: $0.00089 | Saved: 70%
```

Code flow:
```
router.py → expand_query()
  regex "ocean_view" matches → adds ["waterfront","beachfront","seaview","ocean view"]
  regex "penthouse"  matches → adds ["top floor","rooftop","luxury penthouse"]
  returns: ("ocean view penthouse Miami waterfront beachfront ...", ["ocean_view","penthouse"])

embedder.py → search(expanded_query, top_k=15)
  _embed([expanded_query]) → 768-dim vector via OpenAI
  dot product against pre-normalized 50-listing matrix
  argsort descending → top 15 listing dicts returned

embedder.py → get_token_counts(15 listings)
  tiktoken encodes all 20 fields → ~3400 tokens  (full)
  tiktoken encodes 5 fields only → ~1400 tokens  (trimmed)

reranker.py → rerank("ocean view penthouse Miami", 15 listings)
  cross-encoder scores 15 pairs: (query, "title + bedrooms + description")
  sigmoid normalizes each raw score to 0–1
  score > 0.7  → accepted  (~3–5 listings)
  0.15–0.7    → ambiguous (~3–5 listings)
  score < 0.15 → discarded (remainder)

pipeline.py
  context_listings = accepted + ambiguous (~6–8 listings)
  _trim_listing() keeps: id, title, price, bedrooms, amenities, description
  is_complex_query() → False (≤8 words, no financial keywords) → FAST_MODEL
  system_prompt = static block (>1024 tokens) + trimmed listings JSON at bottom
  OpenAI call: gpt-4o-mini, max_tokens=500
  usage.prompt_tokens_details.cached_tokens → 0 (first call, nothing cached yet)
  cost     = ~1200 input × $0.15/1M + ~150 output × $0.60/1M ≈ $0.00027
  baseline = (~3400 + 1200) × $2.50/1M + ~150 × $10.00/1M   ≈ $0.00116
```

---

### 2. `ocean view penthouse Miami` *(repeat)*
**What fires:** prefix cache hits — cached tokens jump from 0 to ~900. Same query, 50% cheaper input.

Terminal output:
```
[LLM     ] Input: 1166 tokens | Cached: 912 | Output: 143
[COST    ] Actual: $0.00019 | Baseline: $0.00089 | Saved: 79%
```

Code flow:
```
router.py   → same rules fire, same expanded query
embedder.py → same 15 listings returned (deterministic cosine sort)
reranker.py → same scores (model is deterministic)

pipeline.py
  system_prompt built identically — static block is byte-for-byte the same
  listings JSON is also the same → entire system prompt is identical
  OpenAI call: gpt-4o-mini
  usage.prompt_tokens_details.cached_tokens → ~900+ (prefix now cached)
  cost = (input - cached) × $0.15/1M + cached × $0.075/1M + output × $0.60/1M
       → noticeably cheaper than turn 1
```

---

### 3. `3 bedroom pool Austin under 700k`
**What fires:** pool routing rule, JSON trimming visible, re-ranker discards non-Austin/non-pool listings hard.

Terminal output:
```
[ROUTER  ] Rules fired: pool → query expanded
[TRIM    ] Full JSON: 3414 tokens → Trimmed: 1461 tokens
[RERANK  ] Accepted: 3 | Ambiguous: 2 | Discarded: 10
```

Code flow:
```
router.py → expand_query()
  regex "pool" matches → adds ["swimming pool","heated pool"]
  "3 bedroom", "Austin", "under 700k" — no rules match
  returns: ("3 bedroom pool Austin under 700k swimming pool heated pool", ["pool"])

embedder.py → search() returns top 15
  query vector skewed toward pool + Austin listings

reranker.py → rerank("3 bedroom pool Austin under 700k", 15 listings)
  specific query → cross-encoder scores more decisively
  listings without pool or Austin score low → higher discarded count than scenario 1

pipeline.py
  context_listings = accepted + ambiguous (fewer listings reach the LLM)
  is_complex_query() → False → gpt-4o-mini
  cached_tokens → high (same static system prompt block, now warm in cache)
  savings % higher than scenario 1 — fewer listings in prompt = fewer tokens
```

---

### 4. `compare investment roi Miami vs Austin`
**What fires:** no routing rules match, model routing escalates to gpt-4o.

Terminal output:
```
[ROUTER  ] No rules fired — query passed through
[MODEL   ] Routing → gpt-4o (complex query)
```

Code flow:
```
router.py → expand_query()
  no amenity rules match this query
  fired_rules = []

pipeline.py
  is_complex_query() checks:
    len("compare investment roi Miami vs Austin".split()) = 7 → not >8
    COMPLEX_KEYWORDS regex matches "compare", "roi", "vs"    → True
  model = SMART_MODEL → gpt-4o
  cost = input × $2.50/1M + output × $10.00/1M → higher than previous queries
  baseline also uses gpt-4o → savings % lower here
  (model routing already chose the expensive model — re-ranker + trimming still save)
```

---

### 5. `show me something affordable in Boston` *(turn 5)*
**What fires:** history compaction — all previous turns summarized into one message.

Terminal output:
```
[COMPACT ] History summarized — 847 tokens saved
```

Code flow:
```
compactor.py → maybe_compact(messages, client, turn_count=5)
  turn_count >= HISTORY_LIMIT (5) and 5 % 5 == 0 → True, compact

  messages at this point:
    [user1, assistant1, user2, assistant2, user3, assistant3,
     user4, assistant4, user5, assistant5]

  last_user    = non_system[-1]   → assistant5 (last message)
  to_summarize = non_system[:-1]  → first 9 messages

  gpt-4o-mini call: "Summarize this conversation in one paragraph"
  summary_message = {"role":"assistant","content":"[Summary]: ..."}

  new_messages = [summary_message, assistant5]
  tokens_saved = tiktoken(9 messages) - tiktoken(summary)

next query (turn 6):
  messages passed in = [summary_message, assistant5]
  context window stays flat regardless of session length
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
