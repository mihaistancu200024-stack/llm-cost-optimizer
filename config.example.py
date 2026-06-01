OPENAI_API_KEY = "sk-..."

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 768

FAST_MODEL = "gpt-4o-mini"
SMART_MODEL = "gpt-4o"

RERANKER_LOW = 0.15
RERANKER_HIGH = 0.70

HISTORY_LIMIT = 5

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "text-embedding-3-small": {"input": 0.02},
}
