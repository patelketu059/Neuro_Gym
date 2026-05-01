from __future__ import annotations
import os
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FASTAPI_URL = os.environ.get('FASTAPI_URL', "http://localhost:8000")

COL_COLORS = {
    "gym_images": "#185FA5",   # blue
    "gym_text":   "#3B6D11",   # green
    "gym_tables": "#854F0B",   # amber
    "bm25":       "#993556",   # pink
}
COL_LABELS = {
    "gym_images": "PDF page",
    "gym_text":   "coaching text",
    "gym_tables": "week data",
    "bm25":       "BM25",
}

RETRIEVAL_CONFIGS = [
    "A — images only",
    "B — text only",
    "C — tables only",
    "D — all dense",
    "E — tables + BM25",
    "F — all + BM25",
    "G — hybrid + rerank",
    "H — BM25 only",
]

CONFIG_DESC = {
    "A — images only":     "PDF pages only",
    "B — text only":       "Coaching text only",
    "C — tables only":     "Week data only",
    "D — all dense":       "All collections, no BM25",
    "E — tables + BM25":   "Tables + BM25 keyword",
    "F — all + BM25":      "All + BM25 (default)",
    "G — hybrid + rerank": "Full hybrid + reranker",
    "H — BM25 only":       "Keyword search only",
}

GOLDEN_QUESTIONS = [
    "What was athlete_00042's squat weight and RPE in week 8?",
    "How many sets and reps did athlete_00117 do on bench press during the intensification phase?",
    "What accessories did athlete_00250 perform on lower body days in week 3?",
    "What is athlete_00089's Dots score and what training level does that correspond to?",
    "What are athlete_00033's peak squat, bench, and deadlift competition numbers?",
    "What primary program did athlete_00178 run and did they have a secondary program?",
    "What happened to athlete_00042's squat load in week 10 — was it a deload week?",
    "What is athlete_00301's bodyweight and which IPF weight class do they compete in?",
    "How did athlete_00117's deadlift progress from week 1 to week 12?",
    "At what point in the 12-week block do most advanced athletes show an RPE spike?",
    "How does athlete_00089's training volume change between accumulation and realisation?",
    "Which week typically has the lowest squat weight in the 12-week block and why?",
    "How does RPE change during a deload week compared to the week before it?",
    "Who are the strongest deadlifters in the advanced category ranked by Dots score?",
    "Compare the peak bench press of an intermediate and an advanced athlete.",
    "Do elite athletes use higher RPE values throughout the block or only in the final weeks?",
    "I am a beginner and my squat has stalled for 3 weeks. What do athletes in the dataset typically do?",
    "How should I structure my accessories on lower body days if I am running a powerlifting block?",
    "My RPE on deadlift is consistently hitting 9.5 in week 6. Is that normal?",
    "What does a good deload week look like for an intermediate lifter?",
    "Can you describe what athlete_00042's progression chart looks like?",
    "Looking at the frequency heatmap, how many days per week does a typical advanced athlete train?",
    "What does athlete_00117's volume radar chart show about upper vs lower body training?",
    "Which athlete's chart shows a clear deload dip followed by a sharp peak?",
    "Which athletes run a 5/3/1 primary program and how does their squat progression compare?",
]

DEFAULT_CONFIG = "F — all + BM25"
MESSAGE_AREA_HEIGHT = 620  # px — fixed-height scrollable history pane


st.set_page_config(
    page_title = "Neuro GYM RAG — Lifting Coach",
    page_icon  = "🏋",
    layout     = "wide",
)


# -----------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------

def _init_state():
    defaults = {
        "session_id":     str(uuid.uuid4())[:8],
        "messages":       [],
        "uploader_key":   0,
        "pdf_paths":      [],
        "pdf_index":      0,
        "config_name":    DEFAULT_CONFIG,
        "prefill_query":  "",
        "uploaded_image": None,
        "pending_prompt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# -----------------------------------------------------------------------
# PDF helpers
# -----------------------------------------------------------------------

PDF_ARCHIVE_DIR = ROOT / "data" / "pdfs" / "pdfs_archive"


def _resolve_pdf_path(pdf_path: str) -> Path:
    """Backend hands us values like 'pdfs/athlete_00042.pdf' or just
    'athlete_00042.pdf'. The on-disk layout is data/pdfs/pdfs_archive/<basename>.
    Strip any directory prefix and join under PDF_ARCHIVE_DIR. Absolute paths
    pass through unchanged."""
    p = Path(pdf_path)
    if p.is_absolute():
        return p
    return PDF_ARCHIVE_DIR / p.name


def _render_pdf_page(pdf_path: str, page: int = 0, dpi: int = 150) -> bytes | None:
    try:
        import fitz
        full_path = _resolve_pdf_path(pdf_path)
        if not full_path.is_file():
            return None
        doc = fitz.open(str(full_path))
        if page >= len(doc):
            page = 0
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page].get_pixmap(matrix=mat)
        return pix.tobytes("png")
    except Exception:
        return None


# -----------------------------------------------------------------------
# Sidebar — session, clear, health
# -----------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Gym RAG")
    st.caption("Powerlifting coaching chatbot")
    st.divider()

    st.markdown(f"**Session** `{st.session_state.session_id}`")
    if st.button("Clear conversation", use_container_width=True):
        try:
            requests.delete(
                f"{FASTAPI_URL}/chat/{st.session_state.session_id}", timeout=5
            )
        except Exception:
            pass
        st.session_state.messages       = []
        st.session_state.pdf_paths      = []
        st.session_state.pdf_index      = 0
        st.session_state.uploaded_image = None
        st.session_state.pending_prompt = None
        st.session_state.session_id     = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()

    try:
        h = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        if h.status_code == 200:
            data   = h.json()
            status = data.get("status", "unknown")
            if status == "ok":
                st.success("API connected")
            else:
                st.warning(f"API {status}")
            for col, count in data.get("qdrant", {}).items():
                if isinstance(count, int):
                    st.caption(f"{col}: {count:,} vectors")
            st.caption(
                "Gemini: ready" if data.get("gemini_loaded") else "Gemini: not configured"
            )
        else:
            st.error("API error")
    except Exception:
        st.error("API unreachable — start FastAPI first")
        st.caption("uvicorn app.main:app --port 8000")


# -----------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------

def _render_sources_block(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources — {len(sources)} results"):
        for src in sources:
            aid   = src.get("athlete_id", "?")
            col   = src.get("collection", "?")
            score = src.get("score", 0.0)
            color = COL_COLORS.get(col, "#888")
            label = COL_LABELS.get(col, col)

            extra = ""
            if src.get("week"):
                extra += f" · week {src['week']}"
            if src.get("block_phase"):
                extra += f" · {src['block_phase']}"
            if src.get("chunk_index") is not None:
                extra += f" · chunk {src['chunk_index']}"
            if src.get("page_number") is not None:
                extra += f" · page {src['page_number']}"

            st.markdown(
                f"<div style='padding:4px 0;border-left:3px solid {color};"
                f"padding-left:8px;margin-bottom:4px'>"
                f"<b>{aid}</b> · "
                f"<span style='color:{color}'>{label}</span>"
                f"{extra} · <code>{score:.3f}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_athlete_profiles(sources: list[dict], athlete_ids: list[str]) -> None:
    if not athlete_ids:
        return
    with st.expander(f"Athlete profiles — {len(athlete_ids)} matched"):
        sources_by_aid = {
            s.get("athlete_id"): s.get("payload", {})
            for s in sources
            if s.get("athlete_id") in athlete_ids
        }
        for aid in athlete_ids:
            p = sources_by_aid.get(aid, {})
            st.markdown(
                f"**{aid}** · {p.get('training_level','?')} · Dots {p.get('dots','?')}  \n"
                f"S {p.get('squat_peak_kg','?')}kg / B {p.get('bench_peak_kg','?')}kg / "
                f"D {p.get('deadlift_peak_kg','?')}kg  \n"
                f"Program: {p.get('primary_program','?')}"
            )
            st.divider()


def _render_assistant_payload(msg: dict) -> None:
    """Render the metadata that lives on an assistant message:
    latency caption, sources expander, athlete-profile expander."""
    if msg.get("retrieval_ms") is not None:
        st.caption(
            f"Retrieval: {msg['retrieval_ms']} ms  ·  "
            f"Generation: {msg['generation_ms']} ms  ·  "
            f"Config: {msg.get('config_name', '?')}"
        )
    _render_sources_block(msg.get("sources", []))
    _render_athlete_profiles(
        msg.get("sources", []), msg.get("athlete_ids", [])
    )


def _call_chat_api(prompt: str, uploaded_image) -> dict:
    """POST to /chat. Returns a normalized result dict so the caller doesn't
    branch on success/failure."""
    fail_payload = {
        "answer": "",
        "sources": [],
        "pdf_paths": [],
        "athlete_ids": [],
        "retrieval_ms": 0,
        "generation_ms": 0,
        "config_used": st.session_state.config_name,
    }
    try:
        files = {}
        data  = {
            "query":       prompt,
            "session_id":  st.session_state.session_id,
            "config_name": st.session_state.config_name,
        }
        if uploaded_image:
            files["image"] = (
                uploaded_image.name,
                uploaded_image.getvalue(),
                uploaded_image.type,
            )
        resp = requests.post(
            f"{FASTAPI_URL}/chat",
            data    = data,
            files   = files or None,
            timeout = 120,
        )
        if resp.status_code == 200:
            r = resp.json()
            return {
                "answer":        r["response"],
                "sources":       r.get("sources", []),
                "pdf_paths":     r.get("pdf_paths", []),
                "athlete_ids":   r.get("athlete_ids", []),
                "retrieval_ms":  r.get("retrieval_ms", 0),
                "generation_ms": r.get("generation_ms", 0),
                "config_used":   r.get("config_name", st.session_state.config_name),
            }
        return {**fail_payload, "answer": f"Error {resp.status_code}: {resp.text}"}
    except requests.exceptions.Timeout:
        return {**fail_payload, "answer": "Request timed out. Try a simpler query or check the API."}
    except Exception as e:  # noqa: BLE001
        return {**fail_payload, "answer": f"Error connecting to API: {e}"}


# -----------------------------------------------------------------------
# Main layout — chat (left) + PDF viewer (right)
# -----------------------------------------------------------------------

chat_col, pdf_col = st.columns([3, 2], gap="large")

# === Chat column =======================================================
with chat_col:
    st.markdown("### NeuroGym Lifting Coach")

    # --- Toolbar (config / golden questions / attach image) ------------
    tb_left, tb_mid, tb_right = st.columns([3, 3, 2], gap="small")

    with tb_left:
        with st.popover(
            "⚙️  " + st.session_state.config_name, use_container_width=True
        ):
            st.markdown("**Retrieval config**")
            for cfg in RETRIEVAL_CONFIGS:
                active = cfg == st.session_state.config_name
                label  = ("✓ " if active else "   ") + cfg
                if st.button(label, key=f"cfg_{cfg}", use_container_width=True):
                    st.session_state.config_name = cfg
                    st.rerun()
                st.caption(CONFIG_DESC.get(cfg, ""))

    with tb_mid:
        with st.popover("📋  Eval questions", use_container_width=True):
            st.markdown("**Golden eval set**")
            for i, q in enumerate(GOLDEN_QUESTIONS):
                short = q[:60] + "…" if len(q) > 60 else q
                if st.button(short, key=f"gq_{i}", use_container_width=True):
                    st.session_state.prefill_query = q
                    st.rerun()

    with tb_right:
        with st.popover("📎  Attach image", use_container_width=True):
            st.markdown("**Upload training chart**")
            f = st.file_uploader(
                "chart",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key=f"uploader_{st.session_state.uploader_key}",
            )
            if f is not None:
                st.session_state.uploaded_image = f
                st.image(f, width="stretch")
            if st.session_state.uploaded_image and st.button(
                "Remove image", use_container_width=True
            ):
                st.session_state.uploaded_image = None
                st.session_state.uploader_key  += 1
                st.rerun()

    # --- Pills row showing active overrides ----------------------------
    pills: list[str] = []
    if st.session_state.config_name != DEFAULT_CONFIG:
        pills.append(f"⚙️ {st.session_state.config_name}")
    if st.session_state.get("uploaded_image"):
        pills.append(f"📎 {st.session_state.uploaded_image.name}")
    if pills:
        st.caption("  ·  ".join(pills))

    # --- Scrollable message history ------------------------------------
    msg_area = st.container(height=MESSAGE_AREA_HEIGHT, border=True)
    with msg_area:
        if not st.session_state.messages:
            st.markdown(
                "<div style='text-align:center;color:#888;padding:40px 20px'>"
                "💬 Start a conversation with the coach.<br>"
                "Ask about athlete data, training trends, or attach a chart."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        _render_assistant_payload(msg)

    # --- Chat input pinned at the bottom of the chat column ------------
    submitted = st.chat_input(
        "Ask about any training/lifting questions...",
        key="chat_input",
    )

    # Pull a one-shot prefill from session_state if the user clicked an
    # eval-question button. Always clear after read so it doesn't fire again.
    prefill = st.session_state.get("prefill_query", "")
    st.session_state.prefill_query = ""
    prompt = submitted or (prefill if not submitted else None)

    # --- New-prompt processing -----------------------------------------
    if prompt:
        # Clear stale PDFs the moment a new query is fired so the right-hand
        # column blanks while the spinner runs rather than displaying the
        # previous query's results.
        st.session_state.pdf_paths = []
        st.session_state.pdf_index = 0

        # Append the user's message immediately so it appears in the
        # history pane while we wait on the API call below.
        st.session_state.messages.append({"role": "user", "content": prompt})

        with msg_area:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Retrieving and generating..."):
                    result = _call_chat_api(
                        prompt, st.session_state.get("uploaded_image")
                    )

                placeholder.markdown(result["answer"])
                # Build the assistant message dict and render its payload
                # so the user sees sources / profiles / latency before rerun.
                assistant_msg = {
                    "role":          "assistant",
                    "content":       result["answer"],
                    "sources":       result["sources"],
                    "athlete_ids":   result["athlete_ids"],
                    "retrieval_ms":  result["retrieval_ms"],
                    "generation_ms": result["generation_ms"],
                    "config_name":   result["config_used"],
                }
                _render_assistant_payload(assistant_msg)

        # Persist the assistant message + update PDF panel state
        st.session_state.messages.append(assistant_msg)
        st.session_state.pdf_paths = (result["pdf_paths"] or [])[:5]
        st.session_state.pdf_index = 0

        st.rerun()


# === PDF column ========================================================
with pdf_col:
    st.markdown("### PDF Viewer")

    pdf_paths = st.session_state.pdf_paths
    if not pdf_paths:
        st.info("Send a query to retrieve athlete PDFs.")
    else:
        n = len(pdf_paths)
        index = st.session_state.pdf_index

        nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
        with nav_left:
            if st.button(
                "←", disabled=(index == 0), use_container_width=True,
                key="pdf_prev",
            ):
                st.session_state.pdf_index = max(0, index - 1)
                st.rerun()

        with nav_mid:
            st.markdown(
                f"<p style='text-align:center;margin:0;padding:6px 0'>"
                f"<b>{pdf_paths[index]}</b><br>"
                f"<small>PDF {index + 1} of {n}</small></p>",
                unsafe_allow_html=True,
            )

        with nav_right:
            if st.button(
                "→", disabled=(index == n - 1), use_container_width=True,
                key="pdf_next",
            ):
                st.session_state.pdf_index = min(n - 1, index + 1)
                st.rerun()

        pdf_full = _resolve_pdf_path(pdf_paths[index])
        n_pages = 1
        try:
            import fitz
            if pdf_full.is_file():
                n_pages = len(fitz.open(str(pdf_full)))
        except Exception:
            pass

        if n_pages > 1:
            page_index = st.slider(
                "Page", 0, n_pages - 1, 0, key=f"page_{index}"
            )
        else:
            page_index = 0
            st.caption(f"Page 1 of {n_pages}")

        png = _render_pdf_page(pdf_paths[index], page=page_index)
        if png:
            st.image(png, width="stretch")
        else:
            st.warning(f"Could not render {pdf_paths[index]}")

        dot_html = "".join(
            f"<span style='display:inline-block;width:8px;height:8px;"
            f"border-radius:50%;background:{'#185FA5' if i == index else '#ccc'};"
            f"margin:0 3px'></span>"
            for i in range(n)
        )
        st.markdown(
            f"<p style='text-align:center;margin-top:8px'>{dot_html}</p>",
            unsafe_allow_html=True,
        )
