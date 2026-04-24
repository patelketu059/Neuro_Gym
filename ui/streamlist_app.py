from __future__ import annotations
import base64
import io
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


st.set_page_config(
    page_title = 'Neuro GYM RAG - Lifting Coach',
    page_icon  = "🏋",
    layout     = "wide",
)



def _init_state():
    defaults = {
        "session_id": str(uuid.uuid4())[:8],
        "messages": [],
        "uploader_key": 0,
        "pdf_paths": [],
        "pdf_index": 0,
        "config_name": "F — all + BM25",
        "prefill_query": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def _render_pdf_page(pdf_path: str, page: int = 0, dpi: int = 150) -> bytes | None:
    try:
        import fitz
        full_path = (ROOT / "data" / "pdfs" / "pdfs_archive" / pdf_path) if not Path(pdf_path).is_absolute() else Path(pdf_path)
        if not full_path.is_file():
            return None
        
        doc = fitz.open(str(full_path))
        if page >= len(doc):
            page = 0
        
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[page].get_pixmap(matrix = mat)
        return pix.tobytes('png')
    except Exception:
        return None
    

with st.sidebar:
    st.markdown('## Gym RAG')
    st.caption("Lifting Coach Chatbot")
    st.divider()

    st.markdown("**Retrieval Config**")
    config_name = st.selectbox(
        "Config",
        RETRIEVAL_CONFIGS,
        index = RETRIEVAL_CONFIGS.index(st.session_state.config_name),
        label_visibility = "collapsed"
    )
    st.session_state.config_name = config_name

    config_descriptions = {
        "A — images only":     "Dense search on PDF pages only",
        "B — text only":       "Dense search on coaching text only",
        "C — tables only":     "Dense search on NL week strings only",
        "D — all dense":       "All 3 collections, no BM25",
        "E — tables + BM25":   "Tables + BM25, RRF fusion",
        "F — all + BM25":      "All collections + BM25, RRF (default)",
        "G — hybrid + rerank": "Full hybrid + cross-encoder reranker",
        "H — BM25 only":       "Keyword search only, no dense",
    }
    st.caption(config_descriptions.get(config_name, ""))
    st.divider()


    st.markdown("**Golden Eval Questions**")
    selected_q = st.selectbox(
        "Load a test question",
        ["-- Select -- "] + GOLDEN_QUESTIONS,
        label_visibility = 'collapsed'
    )

    if selected_q != "-- Select -- ":
        st.session_state.prefill_query = selected_q
        st.rerun()

    st.divider()


    st.markdown("**Upload Training Chart**")
    uploaded_image = st.file_uploader(
        "Chart Image",
        type = ['png', 'jpg', 'jpeg', 'svg'],
        label_visibility = 'collapsed',
        key = f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_image:
        st.image(uploaded_image, use_column_width = True)
        if st.button("Clear Image", use_container_width = True):
            st.session_state.uploader_key += 1
            st.rerun()

    st.divider()


    try: 
        health = requests.get(f"{FASTAPI_URL}/health", timeout = 3)
        if health.status_code == 200:
            data = health.json()
            status = data.get("status", "unknown")
            if status == "ok":
                st.success("API Connected")
            else:
                st.warning(f"API {status}")
            
            for col, count in data.get("qdrant", {}).items():
                if isinstance(count, int):
                    st.caption(f"{col}: {count} vectors")
            
            if data.get("gemini_loaded"):
                st.caption("Gemini: Ready")
            else:
                st.caption("Gemini: Not Configured")

        else:
            st.error("API ERROR")
    except Exception:
        st.error("API Unreachable: Start FastAPI")
        st.caption("uvicorn app.main_day4:app --port 8000")



chat_col, pdf_col = st.columns([3, 2], gaps = 'large')
with pdf_col:
    st.markdown("### PDF Viewer")

    pdf_paths = st.session_state.pdf_paths
    if not pdf_paths:
        st.info("Send a query for retrievel athelete PDFs")
    else:
        n = len(pdf_paths)
        index = st.session_state.pdf_index

        nav_left, nav_mid, nav_right = st.columns([1, 3, 1])
        with nav_left:
            if st.button("←", disabled = (index == 0), 
                         use_container_width = True):
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
            if st.button("→", disabled=(index == n - 1), 
                         use_container_width=True):
                st.session_state.pdf_index = min(n - 1, index + 1)
                st.rerun()

        pdf_full = ROOT / 'data' / 'pdfs' / 'pdf_archive' / pdf_paths[index]
        n_pages = 1
        try: 
            import fitz
            if pdf_full.is_file():
                n_pages = len(fitz.open(str(pdf_full)))
        except Exception:
            pass

        page_index = st.slider("Page", 0, max(0, n_pages - 1), 0, key = [f'page_{index}'])
        png = _render_pdf_page(pdf_paths[index], page = page_index)
        if png:
            st.image(png, use_column_width = True)
        else:
            st.warning(f"Could not render {pdf_paths[index]}")


        dot_html = "".join(
            f"<span style='display:inline-block;width:8px;height:8px;border-radius:50%;"
            f"background:{'#185FA5' if i == idx else '#ccc'};margin:0 3px'></span>"
            for i in range(n)
        )
        st.markdown(
            f"<p style='text-align:center;margin-top:8px'>{dot_html}</p>",
            unsafe_allow_html=True,
        )           


with chat_col:
    st.markdown("### NeuroGym Lifting Coach")

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

            if msg['role'] == 'assistant':

                if msg.get("retrieval_ms") is not None:
                    r_ms = msg['retrieval_ms']
                    g_ms = msg['generation_ms']

                    st.caption(
                        f"Retrieval: {r_ms} ms  |  Generation: {g_ms} ms  "
                        f"|  Config: {msg.get('config_name', '?')}"
                    )

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"Sources -- {len(sources)} results"):
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
                                f"<div style='padding:4px 0;border-left:3px solid {color};padding-left:8px;margin-bottom:4px'>"
                                f"<b>{aid}</b> · "
                                f"<span style='color:{color}'>{label}</span>"
                                f"{extra} · <code>{score:.3f}</code>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                athlete_ids = msg.get("athlete_ids", [])
                if athlete_ids:
                    with st.expander(f"Athlete profiles -- {len(athlete_ids)} matched"):
                        sources_by_aid = {
                            s['athlete_id']: s.get('payload', {}) for s in sources 
                            if s.get('athlete_id') in athlete_ids
                        }

                        for aid in athlete_ids:
                            p = sources_by_aid.get(aid, {})
                            dots  = p.get("dots", "?")
                            level = p.get("training_level", "?")
                            sq    = p.get("squat_peak_kg", "?")
                            be    = p.get("bench_peak_kg", "?")
                            dl    = p.get("deadlift_peak_kg", "?")
                            prog  = p.get("primary_program", "?")
                            st.markdown(
                                f"**{aid}** · {level} · Dots {dots}  \n"
                                f"S {sq}kg / B {be}kg / D {dl}kg  \n"
                                f"Program: {prog}"
                            )
                            st.divider()


    prefill = st.session_state.get('prefill_query', '')
    st.session_state.prefill_query = ""

    prompt = st.chat_input(
        "Ask about any training/lifting questions...",
        key = 'chat_input'
    )

    if not prompt and prefill:
        prompt = prefill
    
    if prompt: 
        st.session_state.messages.append({"role": "user",
                                          "content": prompt})
        
        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_col:
            with st.chat_message('assistant'):
                with st.spinner('Retrieving and Generating...'):
                    try:
                        files = {}
                        data = {
                            "query": prompt,
                            "session_id": st.session_state.session_id,
                            "config_name": st.session_state.config_name
                        }

                        if uploaded_image: 
                            files['image'] = {
                                uploaded_image.name,
                                uploaded_image.getvalue(),
                                uploaded_image.type
                            }
                        
                        resp = requests.post(
                            f"{FASTAPI_URL}/chat",
                            data = data,
                            files = files or None,
                            timeout = 120
                        )

                        if resp.status_code == 200:
                            result        = resp.json()
                            answer        = result["response"]
                            sources       = result.get("sources", [])
                            pdf_paths_new = result.get("pdf_paths", [])
                            athlete_ids   = result.get("athlete_ids", [])
                            retrieval_ms  = result.get("retrieval_ms", 0)
                            generation_ms = result.get("generation_ms", 0)
                            config_used   = result.get("config_name", st.session_state.config_name)
                        else:
                            answer        = f"Error {resp.status_code}: {resp.text}"
                            sources       = []
                            pdf_paths_new = []
                            athlete_ids   = []
                            retrieval_ms  = 0
                            generation_ms = 0
                            config_used   = st.session_state.config_name

                    except requests.exceptions.Timeout:
                        answer        = "Request timed out. Try a simpler query or check the API."
                        sources       = []
                        pdf_paths_new = []
                        athlete_ids   = []
                        retrieval_ms  = 0
                        generation_ms = 0
                        config_used   = st.session_state.config_name
                    except Exception as e:
                        answer        = f"Error connecting to API: {e}"
                        sources       = []
                        pdf_paths_new = []
                        athlete_ids   = []
                        retrieval_ms  = 0
                        generation_ms = 0
                        config_used   = st.session_state.config_name

                st.markdown(answer)
                st.caption(
                    f"Retrieval: {retrieval_ms} ms  |  Generation: {generation_ms} ms  "
                    f"|  Config: {config_used}"
                )
                

                if sources:
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
                            st.markdown(
                                f"<div style='padding:4px 0;border-left:3px solid {color};padding-left:8px;margin-bottom:4px'>"
                                f"<b>{aid}</b> · "
                                f"<span style='color:{color}'>{label}</span>"
                                f"{extra} · <code>{score:.3f}</code>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                if athlete_ids:
                    with st.expander(f"Athlete profiles — {len(athlete_ids)} matched"):
                        sources_by_aid = {s["athlete_id"]: s.get("payload", {}) for s in sources if s.get("athlete_id")}
                        for aid in athlete_ids:
                            p = sources_by_aid.get(aid, {})
                            st.markdown(
                                f"**{aid}** · {p.get('training_level','?')} · Dots {p.get('dots','?')}  \n"
                                f"S {p.get('squat_peak_kg','?')}kg / B {p.get('bench_peak_kg','?')}kg / D {p.get('deadlift_peak_kg','?')}kg  \n"
                                f"Program: {p.get('primary_program','?')}"
                            )
                            st.divider()

        # Update PDF viewer
        if pdf_paths_new:
            st.session_state.pdf_paths = pdf_paths_new[:5]
            st.session_state.pdf_index = 0

        st.session_state.messages.append({
            "role":          "assistant",
            "content":       answer,
            "sources":       sources,
            "athlete_ids":   athlete_ids,
            "retrieval_ms":  retrieval_ms,
            "generation_ms": generation_ms,
            "config_name":   config_used,
        })

        st.rerun()