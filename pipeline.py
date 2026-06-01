import json

import openai

from compactor import maybe_compact
from config import FAST_MODEL, OPENAI_API_KEY, PRICING, SMART_MODEL
from embedder import Embedder
from reranker import Reranker
from router import expand_query, is_complex_query

client = openai.OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT_TEMPLATE = """\
You are a helpful real estate assistant helping users find their perfect property.
You have deep expertise in real estate markets, property valuation, neighborhood analysis,
and matching properties to buyer preferences.

INSTRUCTIONS FOR RESPONDING:
- Always be specific about which properties match the user's criteria
- Mention price, bedrooms, and key amenities when discussing properties
- If asked about neighborhoods, provide context about lifestyle and commute
- Format property listings clearly with title, price, and key features
- If multiple properties match, rank them by relevance to the user's stated preferences
- Be concise but thorough — buyers need accurate information to make decisions
- Always mention the neighborhood when recommending properties
- If the user's budget isn't clear, present options across different price ranges
- Highlight standout features that differentiate each property
- If a property has ocean views or waterfront access, always mention this prominently

PROPERTY CONTEXT GUIDELINES:
- Luxury properties (above $1.5M) should be discussed with appropriate prestige framing
- Budget-conscious buyers deserve clear value propositions
- Always consider proximity to amenities like schools, transit, and shopping when relevant
- New construction vs. older properties have different maintenance considerations
- HOA fees significantly impact total monthly cost — always factor these in
- Investment properties should be evaluated differently than primary residences
- Penthouse units typically offer the best views and premium finishes
- Waterfront and ocean view properties command premium prices for good reason
- Properties with pools and gyms are especially valuable in warm climates
- Garage and parking availability is a major factor in urban markets

ADDITIONAL GUIDANCE FOR PROPERTY EVALUATION:
- When comparing similar properties, always note the price-per-square-foot if available
- School district quality is one of the most searched criteria for families — mention it
- Walk score and transit access are important for urban buyers
- Pet-friendly buildings and outdoor space matter for buyers with pets
- Storage space, closets, and garage size are practical but often overlooked factors
- Natural light and window orientation (south-facing is desirable) adds real value
- Condo vs. co-op distinctions matter in certain markets — flag this when relevant
- Green features (solar panels, EV chargers, LEED certification) are increasingly valued
- Short-term rental restrictions in HOAs affect investment property suitability
- Historic districts may impose renovation restrictions — mention when relevant

MARKET AND PRICING CONTEXT:
- In competitive markets, properties at or below market value move quickly
- Price reductions signal motivated sellers and negotiation opportunities
- Days-on-market is a useful signal of demand and pricing accuracy
- Cash offers are preferred in fast-moving markets — clarify financing options
- Contingencies (inspection, financing, appraisal) affect offer competitiveness
- Seasonal patterns affect inventory — spring typically has more listings
- Local market conditions (buyer's vs. seller's market) affect negotiation leverage
- New development pipelines can affect future neighborhood values
- Zoning changes and city planning decisions affect long-term investment value
- Crime statistics and safety perception affect desirability and resale value

LIFESTYLE AND PREFERENCE MATCHING:
- Match property style to stated lifestyle (urban vs. suburban vs. rural)
- Commute time and transportation links are critical for working buyers
- Proximity to preferred restaurants, cafes, and entertainment matters
- For remote workers, home office space and internet infrastructure are priorities
- Outdoor enthusiasts value proximity to parks, trails, hiking, or beaches
- Young professionals often prioritize walkability and nightlife access
- Retirees often prioritize single-floor living, low maintenance, and medical access
- Growing families need room to expand — flexible bonus rooms matter
- Empty-nesters may want to downsize but retain quality and location
- First-time buyers need clear guidance on hidden costs and realistic expectations

RESPONSE FORMAT:
- Start with a brief summary of what you found
- List matching properties with: Title | Price | Beds | Key Amenities
- End with a recommendation or next steps

AVAILABLE PROPERTIES:
{listings_json}
"""


def _trim_listing(listing: dict) -> dict:
    return {
        "id": listing.get("id"),
        "title": listing.get("title"),
        "price": listing.get("price"),
        "bedrooms": listing.get("bedrooms"),
        "amenities": listing.get("amenities"),
        "description": listing.get("description"),
    }


def _calculate_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    pricing = PRICING[model]
    non_cached = input_tokens - cached_tokens
    cost = (
        non_cached * pricing["input"] / 1_000_000
        + cached_tokens * pricing.get("cached_input", pricing["input"]) / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
    return cost


def _log(tag: str, msg: str) -> None:
    print(f"\033[36m[{tag:<8}]\033[0m {msg}", flush=True)


def search(
    query: str,
    messages: list[dict],
    turn_count: int,
    embedder: Embedder,
    reranker: Reranker,
) -> dict:
    print(flush=True)
    _log("QUERY", f'"{query}"')

    # Step 1: Router
    expanded_query, fired_rules = expand_query(query)
    if fired_rules:
        _log("ROUTER", f"Rules fired: {', '.join(fired_rules)} → query expanded")
    else:
        _log("ROUTER", "No rules fired — query passed through")

    # Step 2: Embed + Search
    retrieved_listings = embedder.search(expanded_query, top_k=15)
    _log("EMBED", f"Top {len(retrieved_listings)} listings retrieved")

    # Step 3: Token counting
    token_counts = embedder.get_token_counts(retrieved_listings)
    _log("TRIM", f"Full JSON: {token_counts['full_tokens']} tokens → Trimmed: {token_counts['trimmed_tokens']} tokens")

    # Step 4: Re-rank
    rerank_result = reranker.rerank(query, retrieved_listings)
    accepted = rerank_result["accepted"]
    ambiguous = rerank_result["ambiguous"]
    discarded = rerank_result["discarded"]
    _log("RERANK", f"Accepted: {len(accepted)} | Ambiguous: {len(ambiguous)} | Discarded: {len(discarded)}")

    reranker_stats = {
        "accepted": len(accepted),
        "ambiguous": len(ambiguous),
        "discarded": len(discarded),
    }

    # Step 5: Prepare context
    context_listings = accepted + ambiguous
    if not context_listings:
        context_listings = accepted[:3] if accepted else retrieved_listings[:3]

    trimmed = [_trim_listing(l) for l in context_listings]
    listings_json = json.dumps(trimmed, indent=2)

    # Step 6: Model routing
    model = SMART_MODEL if is_complex_query(query) else FAST_MODEL
    _log("MODEL", f"Routing → {model} ({'complex' if model == SMART_MODEL else 'simple'} query)")

    # Step 7: Build system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(listings_json=listings_json)

    # Step 8: Add user message and call OpenAI
    messages = messages + [{"role": "user", "content": query}]
    _log("LLM", "Calling OpenAI...")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        max_tokens=500,
    )

    answer = response.choices[0].message.content.strip()

    # Step 9: Capture usage
    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    try:
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
    except AttributeError:
        cached_tokens = 0

    _log("LLM", f"Input: {input_tokens} tokens | Cached: {cached_tokens} | Output: {output_tokens}")

    # Step 10: Calculate cost
    cost_usd = _calculate_cost(model, input_tokens, output_tokens, cached_tokens)

    unoptimized_input = token_counts["full_tokens"] + 1200
    unoptimized_cost = (
        unoptimized_input * PRICING["gpt-4o"]["input"] / 1_000_000
        + output_tokens * PRICING["gpt-4o"]["output"] / 1_000_000
    )

    _log("COST", f"Actual: ${cost_usd:.5f} | Baseline: ${unoptimized_cost:.5f} | Saved: {((unoptimized_cost - cost_usd) / unoptimized_cost * 100) if unoptimized_cost > 0 else 0:.0f}%")

    llm_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cost_usd": cost_usd,
    }

    # Step 11: Append assistant response
    messages = messages + [{"role": "assistant", "content": answer}]

    # Step 12: Compact
    messages, was_compacted, tokens_saved = maybe_compact(messages, client, turn_count)
    if was_compacted:
        _log("COMPACT", f"History summarized — {tokens_saved} tokens saved")

    return {
        "answer": answer,
        "model_used": model,
        "fired_rules": fired_rules,
        "token_counts": token_counts,
        "reranker_stats": reranker_stats,
        "llm_usage": llm_usage,
        "unoptimized_cost_usd": unoptimized_cost,
        "savings_usd": unoptimized_cost - cost_usd,
        "messages": messages,
        "was_compacted": was_compacted,
        "tokens_saved_compaction": tokens_saved,
    }
