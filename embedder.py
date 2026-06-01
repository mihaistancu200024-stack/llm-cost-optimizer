import json
import numpy as np
import tiktoken
import openai

from config import OPENAI_API_KEY, EMBED_MODEL, EMBED_DIMS

_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def _embed(texts: list[str]) -> np.ndarray:
    response = _client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIMS,
    )
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def _listing_text(listing: dict) -> str:
    parts = [
        listing.get("title", "").strip(),
        listing.get("neighborhood", "").strip(),
        " ".join(listing.get("amenities", [])),
        listing.get("description", "").strip(),
    ]
    return " ".join(p for p in parts if p)


class Embedder:
    def __init__(self):
        with open("data/listings.json", "r", encoding="utf-8") as f:
            self._listings: list[dict] = json.load(f)

        texts = [_listing_text(l) for l in self._listings]
        self._embeddings = _embed(texts)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings_normed = self._embeddings / np.maximum(norms, 1e-10)

    def search(self, query: str, top_k: int = 15) -> list[dict]:
        q_vec = _embed([query])[0]
        q_norm = q_vec / max(np.linalg.norm(q_vec), 1e-10)
        scores = self._embeddings_normed @ q_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self._listings[i] for i in top_indices]

    def get_token_counts(self, listings: list[dict]) -> dict:
        enc = tiktoken.get_encoding("cl100k_base")

        full_text = json.dumps(listings)
        full_tokens = len(enc.encode(full_text))

        trimmed = [
            {
                "title": l.get("title"),
                "price": l.get("price"),
                "bedrooms": l.get("bedrooms"),
                "amenities": l.get("amenities"),
                "description": l.get("description"),
            }
            for l in listings
        ]
        trimmed_text = json.dumps(trimmed)
        trimmed_tokens = len(enc.encode(trimmed_text))

        return {"full_tokens": full_tokens, "trimmed_tokens": trimmed_tokens}
