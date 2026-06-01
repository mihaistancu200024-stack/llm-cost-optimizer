import streamlit as st

import pipeline
from embedder import Embedder
from reranker import Reranker

st.set_page_config(page_title="PropSearch — Cost-Optimized RAG", layout="wide")


@st.cache_resource
def load_models():
    return Embedder(), Reranker()


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "total_unoptimized_cost_usd": 0.0,
            "total_savings_usd": 0.0,
            "routing_saves": 0,
            "reranker_discarded": 0,
            "reranker_ambiguous_sent": 0,
            "compaction_count": 0,
        }


init_session()
embedder, reranker = load_models()

col_chat, col_sidebar = st.columns([3, 1])

# ---------- LEFT: Chat ----------
with col_chat:
    st.title("PropSearch")
    st.caption("Cost-Optimized Real Estate RAG")

    for entry in st.session_state.chat_display:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            meta = entry.get("meta")
            if meta and entry["role"] == "assistant":
                with st.expander("Cost details"):
                    st.markdown(f"**Model:** `{meta['model_used']}`")
                    usage = meta["llm_usage"]
                    st.markdown(
                        f"**Tokens** — Input: {usage['input_tokens']} | "
                        f"Output: {usage['output_tokens']} | "
                        f"Cached: {usage['cached_tokens']}"
                    )
                    unopt_call = meta.get("unoptimized_cost_usd", 0)
                    actual_call = usage["cost_usd"]
                    saved_call = meta.get("savings_usd", 0)
                    call_pct = (saved_call / unopt_call * 100) if unopt_call > 0 else 0
                    st.markdown(
                        f"**Cost** — Baseline: `${unopt_call:.5f}` → "
                        f"Actual: `${actual_call:.5f}` → "
                        f"**Saved: ${saved_call:.5f} ({call_pct:.0f}%)**"
                    )

                    rs = meta["reranker_stats"]
                    st.markdown(
                        f"**Re-ranker** — Accepted: {rs['accepted']} | "
                        f"Ambiguous: {rs['ambiguous']} | "
                        f"Discarded: {rs['discarded']}"
                    )

                    tc = meta["token_counts"]
                    st.markdown(
                        f"**JSON trimming** — Full: {tc.get('full_tokens', 'N/A')} tokens → "
                        f"Trimmed: {tc.get('trimmed_tokens', 'N/A')} tokens"
                    )

                    fired = meta["fired_rules"]
                    if fired:
                        st.markdown(f"**Routing rules fired:** {', '.join(fired)}")
                    else:
                        st.markdown("**Routing:** no rules fired (pass-through)")

                    if meta.get("was_compacted"):
                        st.markdown(
                            f"**History compacted** — tokens saved: {meta['tokens_saved_compaction']}"
                        )

    query = st.chat_input("Search for properties...")

    if query:
        st.session_state.chat_display.append({"role": "user", "content": query, "meta": None})
        st.session_state.turn_count += 1

        with st.spinner("Searching properties..."):
            result = pipeline.search(
                query=query,
                messages=st.session_state.messages,
                turn_count=st.session_state.turn_count,
                embedder=embedder,
                reranker=reranker,
            )

        st.session_state.messages = result["messages"]

        stats = st.session_state.session_stats
        usage = result["llm_usage"]
        stats["total_input_tokens"] += usage["input_tokens"]
        stats["total_output_tokens"] += usage["output_tokens"]
        stats["total_cached_tokens"] += usage["cached_tokens"]
        stats["total_cost_usd"] += usage["cost_usd"]
        stats["total_unoptimized_cost_usd"] += result["unoptimized_cost_usd"]
        stats["total_savings_usd"] += result["savings_usd"]
        stats["reranker_discarded"] += result["reranker_stats"]["discarded"]
        stats["reranker_ambiguous_sent"] += result["reranker_stats"]["ambiguous"]
        if result["fired_rules"]:
            stats["routing_saves"] += 1
        if result["was_compacted"]:
            stats["compaction_count"] += 1

        st.session_state.chat_display.append({
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "model_used": result["model_used"],
                "llm_usage": usage,
                "unoptimized_cost_usd": result["unoptimized_cost_usd"],
                "savings_usd": result["savings_usd"],
                "reranker_stats": result["reranker_stats"],
                "token_counts": result["token_counts"],
                "fired_rules": result["fired_rules"],
                "was_compacted": result["was_compacted"],
                "tokens_saved_compaction": result["tokens_saved_compaction"],
            },
        })

        st.rerun()

# ---------- RIGHT: Sidebar ----------
with col_sidebar:
    st.header("Session Cost Dashboard")

    stats = st.session_state.session_stats
    total_tokens = stats["total_input_tokens"] + stats["total_output_tokens"]

    unopt = stats["total_unoptimized_cost_usd"]
    actual = stats["total_cost_usd"]
    saved = stats["total_savings_usd"]
    pct = (saved / unopt * 100) if unopt > 0 else 0

    st.metric("Baseline (no optimizations)", f"${unopt:.5f}")
    st.metric("Actual cost (optimized)", f"${actual:.5f}")
    st.metric("You saved", f"${saved:.5f}", delta=f"-{pct:.0f}% vs baseline")

    st.divider()

    st.metric("Total tokens used", f"{total_tokens:,}")
    st.metric("Cached tokens (prefix cache)", f"{stats['total_cached_tokens']:,}")
    st.metric("Listings filtered by re-ranker", stats["reranker_discarded"])
    st.metric("LLM calls replaced by code (routing)", stats["routing_saves"])
    st.metric("History compactions", stats["compaction_count"])

    with st.expander("How it works"):
        st.markdown(
            "**Query routing** — Rule-based expansion runs before embedding, "
            "so simple queries never hit the LLM unnecessarily."
        )
        st.markdown(
            "**Semantic search** — Embeddings retrieve the top-15 candidates "
            "from the vector store at a fraction of the cost of full-text LLM search."
        )
        st.markdown(
            "**Re-ranking** — A lightweight scorer filters out irrelevant listings "
            "before they reach the expensive LLM context window."
        )
        st.markdown(
            "**JSON trimming** — Only title, price, bedrooms, amenities, and description "
            "are sent to the LLM; raw listing objects can be 3-5× larger."
        )
        st.markdown(
            "**Prefix caching** — The system prompt is ≥1024 tokens so OpenAI can cache "
            "it across turns, cutting input token costs by up to 50%."
        )
        st.markdown(
            "**Smart/fast routing** — Simple queries use gpt-4o-mini (17× cheaper input); "
            "complex queries escalate to gpt-4o automatically."
        )
        st.markdown(
            "**History compaction** — Every 5 turns, old conversation turns are summarised "
            "into a single message to keep the context window lean."
        )
