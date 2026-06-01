import contextlib
import io
import logging
import math
import os

from sentence_transformers import CrossEncoder
from transformers import logging as transformers_logging

from config import RERANKER_LOW, RERANKER_HIGH

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()

with contextlib.redirect_stderr(io.StringIO()):
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class Reranker:
    def __init__(self):
        self._model = _cross_encoder

    def rerank(self, query: str, listings: list[dict]) -> dict:
        pairs = [
            (query, f"{listing.get('title', '')} {listing.get('bedrooms', '')} bedrooms {listing.get('description', '')}")
            for listing in listings
        ]
        raw_scores = self._model.predict(pairs)

        accepted = []
        ambiguous = []
        discarded = []
        scores = {}

        for listing, raw in zip(listings, raw_scores):
            listing_id = listing.get("id", listing.get("title", ""))
            score = 1.0 / (1.0 + math.exp(-float(raw)))
            scores[listing_id] = score

            if score < RERANKER_LOW:
                discarded.append(listing)
            elif score > RERANKER_HIGH:
                accepted.append(listing)
            else:
                ambiguous.append(listing)

        return {
            "accepted": accepted,
            "ambiguous": ambiguous,
            "discarded": discarded,
            "scores": scores,
        }
