# app/app.py
#
# Purpose:
#   Minimal Streamlit app for:
#     - multi-turn chat (left panel)
#     - retrieval debug panel (right panel)
#     - BigQuery logging of each user turn
#
# Dependencies:
#   - src/rag_retrieve.py provides retrieve()
#   - src/bq_log_event.py provides log_event()
#
# How it works:
#   Streamlit reruns this script on every interaction.
#   We use st.session_state to persist chat history + latest retrieval results.

import sys
from pathlib import Path

# Add the project root to Python path so "import src..." works under Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uuid  # Generate a unique session_id for the user session.
import streamlit as st  # Streamlit UI framework.

# Import retrieval function (local FAISS + embeddings).
from src.rag_retrieve import retrieve

# Import BigQuery logger for events.
from src.bq_log_event import log_event

from src.rag_answer_local import answer_from_retrieval
from src.bq_log_event_v2 import log_event_v2

# -----------------------------
# Page config (do this before other Streamlit calls)
# -----------------------------

st.set_page_config(page_title="RAG Eval Lab", layout="wide")


# -----------------------------
# Initialize session state (persists across reruns)
# -----------------------------

# session_id: a stable ID for this browser session (used for BigQuery logs).
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# messages: list of chat messages, each message is a dict {"role": "user"/"assistant", "content": "..."}.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# turn_id: simple counter for user turns (increment each time user sends a message).
if "turn_id" not in st.session_state:
    st.session_state["turn_id"] = 0

# latest_retrieval: store last retrieval payload so we can show it in the right panel.
if "latest_retrieval" not in st.session_state:
    st.session_state["latest_retrieval"] = None

# latest_query: store last user query string for the debug panel.
if "latest_query" not in st.session_state:
    st.session_state["latest_query"] = ""


# -----------------------------
# App title
# -----------------------------

st.title("RAG Eval Lab — Chat + Retrieval Debugger")


# -----------------------------
# Layout: two columns (left=chat, right=debugger)
# -----------------------------

left_col, right_col = st.columns([2, 1])  # Left column wider than right column.


# -----------------------------
# LEFT: Chat UI
# -----------------------------
with left_col:
    st.subheader("Chat")

    # Render existing chat messages from session_state.
    for msg in st.session_state["messages"]:
        # st.chat_message renders a message bubble for role="user" or "assistant".
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box at the bottom of the chat panel.
    user_text = st.chat_input("Type a question and press Enter...")

    # When the user submits text, Streamlit reruns and user_text will be non-empty.
    if user_text:
        # 1) Append user message to history.
        st.session_state["messages"].append({"role": "user", "content": user_text})

        # 2) Increment the turn counter (this is the ID for this user query).
        st.session_state["turn_id"] += 1
        turn_id = st.session_state["turn_id"]

        # 3) Run retrieval.
        #    - k=5 results
        #    - dedupe_field doc_id (can switch to title later)
        #    - candidate_multiplier=5 for diversity
        retrieval_payload = retrieve(
            query=user_text,
            k=5,
            dedupe_field="doc_id",
            candidate_multiplier=5,
        )
        # Generate a deterministic grounded answer from retrieved sources (no LLM).
        top_score = retrieval_payload["results"][0]["score"] if retrieval_payload["results"] else None
        ans = answer_from_retrieval(user_text, retrieval_payload["results"], top_score=top_score)

        # Build a compact citations line for display.
        # Format: [doc_id:chunk_id]
        citeline = "Citations: " + ", ".join(
                [f"[{c['doc_id']}:{c['chunk_id']}]" for c in ans["citations"]]
        ) if ans["citations"] else "Citations: (none)"

        # 4) Store retrieval in session_state so the right panel can display it.
        st.session_state["latest_retrieval"] = retrieval_payload
        st.session_state["latest_query"] = user_text

        # 5) Minimal assistant response for Day 6:
        #    We are not generating a full answer yet; we just confirm retrieval happened.
        #assistant_text = (
        #    f"I retrieved {len(retrieval_payload['results'])} sources "
        #    f"(latency ~{retrieval_payload['latency_ms']} ms). "
        #    "See the Retrieval Debugger on the right."
        #)
        #st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        assistant_text = ans["answer"] + "\n\n" + citeline
        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

        # 6) Log this turn to BigQuery.
        #    We store only IDs + latency (not full contexts) to keep logs small.
        retrieved_chunk_ids = [r["chunk_id"] for r in retrieval_payload["results"]]
        retrieved_doc_ids = [r["doc_id"] for r in retrieval_payload["results"]]

        # Write one row into BigQuery rag_events.
        log_event(
            session_id=st.session_state["session_id"],
            turn_id=turn_id,
            query=user_text,
            dedupe_field=retrieval_payload["dedupe_field"],
            top_k=len(retrieval_payload["results"]),
            candidate_multiplier=5,
            k_candidates=retrieval_payload["k_candidates"],
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            latency_ms=retrieval_payload["latency_ms"],
        )

        # Log v2 telemetry (answer + citation signal).
        log_event_v2(
            session_id=st.session_state["session_id"],
            turn_id=turn_id,
            query=user_text,
            dedupe_field=retrieval_payload["dedupe_field"],
            top_k=len(retrieval_payload["results"]),
            candidate_multiplier=5,
            k_candidates=retrieval_payload["k_candidates"],
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_doc_ids=retrieved_doc_ids,
            latency_ms=retrieval_payload["latency_ms"],
            answer_len=len(ans["answer"]),
            has_citations=(len(ans["citations"]) > 0),
            n_citations=len(ans["citations"]),
            refusal=bool(ans["refusal"]),
        )
        # 7) Force rerun so the newly appended messages appear immediately.
        st.rerun()


# -----------------------------
# RIGHT: Retrieval Debug Panel
# -----------------------------
with right_col:
    st.subheader("Retrieval Debugger")

    # Show session info (useful when debugging logs).
    st.caption(f"session_id: {st.session_state['session_id']}")

    # If no retrieval has happened yet, show instructions.
    if st.session_state["latest_retrieval"] is None:
        st.info("Ask a question in the chat to see retrieved chunks here.")
    else:
        # Display the last query and retrieval stats.
        st.write(f"**Latest query:** {st.session_state['latest_query']}")
        st.write(f"**Latency:** {st.session_state['latest_retrieval']['latency_ms']} ms")
        st.write(f"**Candidates searched:** {st.session_state['latest_retrieval']['k_candidates']}")
        st.write(f"**Dedupe field:** {st.session_state['latest_retrieval']['dedupe_field']}")

        st.divider()

        # Display each retrieved result.
        for i, r in enumerate(st.session_state["latest_retrieval"]["results"], start=1):
            st.write(f"**[{i}] score={r['score']:.4f}**  doc_id={r['doc_id']}")
            st.caption(f"chunk_id={r['chunk_id']}  title={r.get('title','')}")
            st.write(r["text_preview"])
            st.divider()
