import streamlit as st
from app.rag import answer
from app.prompts import MANDATORY_CITATION_LINE

st.set_page_config(page_title="InfoHub RAG", page_icon="📚", layout="wide")
st.title("📚 InfoHub RAG (Demo)")
st.caption("Agent answers in Georgian and always includes the mandatory InfoHub citation line.")

with st.expander("Mandatory citation line (always included)", expanded=False):
    st.code(MANDATORY_CITATION_LINE, language="text")

if "history" not in st.session_state:
    st.session_state.history = []

q = st.text_input("Ask a question (any language). Answer will be in Georgian:", placeholder="e.g. VAT rules...")

col1, col2 = st.columns([1, 4])
with col1:
    k = st.number_input("Top-K", min_value=3, max_value=12, value=6, step=1)
with col2:
    run = st.button("Ask")

if run and q.strip():
    res = answer(q.strip(), k=int(k))
    st.session_state.history.append((q.strip(), res))

for user_q, res in reversed(st.session_state.history):
    st.markdown("### Question")
    st.write(user_q)
    st.markdown("### Answer")
    st.write(res["answer"])
    with st.expander("Sources (raw metadata)", expanded=False):
        st.json(res["sources"])
    st.divider()
