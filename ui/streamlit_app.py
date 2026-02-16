from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def _secrets_to_env() -> None:
    """
    On Streamlit Cloud: st.secrets exists.
    Locally: if you have .streamlit/secrets.toml, it also exists.
    This maps secrets into environment variables so app.settings can read them.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:
        return

    for k, v in secrets.items():
        if isinstance(v, (str, int, float, bool)):
            os.environ.setdefault(k, str(v))


_secrets_to_env()


from app.bootstrap_index import ensure_chroma_index  # noqa: E402

ok = ensure_chroma_index()
chroma_dir = Path(os.getenv("CHROMA_DIR", "./data/index"))
db_path = chroma_dir / "chroma.sqlite3"

if not ok or not db_path.exists():
    st.set_page_config(page_title="InfoHub RAG", page_icon="📚", layout="wide")
    st.title("📚 InfoHub RAG (Demo)")
    st.error(
        "ინდექსი ვერ ჩაიტვირთა.\n\n"
        "შეამოწმეთ:\n"
        "- INDEX_URL (GitHub Release zip)\n"
        "- CHROMA_DIR (მაგ: ./data/index)\n\n"
        f"მოსალოდნელი ფაილი: {db_path}"
    )
    st.stop()

# Optional: show DB size so you can confirm it’s not the tiny empty DB
st.set_page_config(page_title="InfoHub RAG", page_icon="📚", layout="wide")
st.title("📚 InfoHub RAG (Demo)")
st.caption(f"✅ Index ready: {db_path} ({db_path.stat().st_size} bytes)")

from app.rag import answer  # noqa: E402
from app.prompts import MANDATORY_CITATION_LINE  # noqa: E402

st.info(
    "ℹ️ **შენიშვნა:** ეს დემო მუშაობს InfoHub-ის დოკუმენტების *ინდექსირებულ* ნაწილზე "
    "(ამ ეტაპზე **დაახლოებით 500 დოკუმენტი**). ამიტომ შესაძლებელია, რომ ზოგიერთი კითხვა "
    "ამ ინდექსში არ მოიძებნოს ან არასაკმარისად იყოს დაფარული.\n\n"
    "სატესტო თემები (რაც მინიმალურად გატესტილი მაქვს): **იარაღი / ნარკოტიკი / ფეთქებადი მასალები / დეკლარაცია / საბაჟო**."
)

st.caption("Agent answers in Georgian and always includes the mandatory InfoHub citation line.")

with st.expander("Mandatory citation line (always included)", expanded=False):
    st.code(MANDATORY_CITATION_LINE, language="text")

colA, colB = st.columns([1, 4])
with colA:
    if st.button("Clear chat"):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input(
    "Ask a question (any language). Answer will be in Georgian:",
    placeholder="e.g. VAT refund rules...",
)

col1, col2 = st.columns([1, 4])
with col1:
    k = st.number_input("Top-K (რეკომენდირებული: 12)", min_value=6, max_value=12, value=12, step=1)
with col2:
    run = st.button("Ask")

if run and q.strip():
    res = answer(q.strip(), k=int(k))
    st.session_state.history.append((q.strip(), res))


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.write("წყაროები: (რელევანტური წყაროები ვერ მოიძებნა)")
        return

    for s in sources:
        title = s.get("title") or "Untitled"
        url = s.get("url") or ""
        page = s.get("page", None)
        suffix = f" — გვერდი {page}" if page is not None else ""
        st.markdown(f"- **{title}** — {url}{suffix}")


for user_q, res in reversed(st.session_state.history):
    st.markdown("### Question")
    st.write(user_q)

    meta = (res.get("meta", {}) or {})
    provider = meta.get("provider")
    model_used = meta.get("model_used")
    fallback_used = meta.get("fallback_used")
    k_used = meta.get("k")

    st.markdown("### Answer")
    st.write(res.get("answer", ""))

    with st.expander("Run metadata", expanded=False):
        st.write(
            {
                "provider": provider,
                "model_used": model_used,
                "fallback_used": fallback_used,
                "k": k_used,
            }
        )

    with st.expander("Sources", expanded=True):
        render_sources(res.get("sources", []) or [])

    st.divider()
