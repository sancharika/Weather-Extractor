# app.py
import streamlit as st

# LangChain + Groq + Vector DB imports
from langchain.text_splitter import RecursiveCharacterTextSplitter

# RAG imports
from rag import run_rag_extraction

# -----------------------
#  Config / Keys
# -----------------------
# Provide in UI or environment
DEFAULT_GROQ_API_KEY = "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"
DEFAULT_GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"

# -----------------------
#  Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Weather Extractor", layout="wide")
st.title("🌦️ RAG Weather Extractor — Scrape, Clean, Embed, Retrieve, Parse")

col1, col2 = st.columns([2,1])

with col1:
    url = st.text_input("Enter weather URL", "https://www.timeanddate.com/weather/india/delhi/ext")
    use_selenium = st.checkbox("Use Selenium if requests fails (requires chromedriver)", value=False)
    groq_api_key = st.text_input("Groq API Key", value=DEFAULT_GROQ_API_KEY, type="password")
    groq_model = st.text_input("Groq Model", value=DEFAULT_GROQ_MODEL)
    chunk_size = st.number_input("Chunk size (characters)", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=200, step=50)
    max_chunks = st.number_input("Max chunks to embed", value=8, min_value=1, max_value=32, step=1)

    if st.button("Extract forecast (RAG)"):
        if not groq_api_key:
            st.error("Groq API key required — enter it above.")
        else:
            with st.spinner("Scraping → cleaning → embedding → retrieving..."):
                try:
                    # update text splitter behavior with UI inputs
                    # re-create chunk_text with new chunk settings by monkeypatching function closure:
                    def chunk_text_local(text: str):
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                            separators=["\n\n", "\n", " ", ""]
                        )
                        return splitter.split_text(text)
                    # inject local chunker
                    globals()["chunk_text"] = chunk_text_local

                    parsed, raw_output = run_rag_extraction(
                        url=url,
                        groq_api_key=groq_api_key,
                        groq_model=groq_model,
                        use_selenium=use_selenium,
                        max_chunks=int(max_chunks)
                    )

                    if parsed is not None:
                        st.success("✅ Parsed JSON forecast (RAG)")
                        st.json(parsed)
                        st.write("### Raw LLM output (for debugging):")
                        st.code(raw_output[:10000])
                    else:
                        st.warning("LLM did not produce clean JSON. See raw output below.")
                        st.write("### Raw LLM output:")
                        st.code(raw_output[:20000])
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

with col2:
    st.markdown("### Quick tips")
    st.markdown("""
- Use **Selenium** for JS-heavy pages (toggle the checkbox). Make sure chromedriver is installed.
- If results are noisy, increase `max_chunks` or chunk size.
- You can swap HuggingFace embedding to a cloud embedding provider if you prefer.
- The pipeline stores vectors in-memory (Chroma) and DOES NOT persist to disk by default.
""")
    st.markdown("### Dependencies (pip)")
    st.code("""
pip install streamlit bs4 requests selenium selenium-stealth chromadb langchain langchain_groq pydantic huggingface-hub sentence-transformers
    """)
    st.markdown("### Notes")
    st.markdown("""
- You must provide a working Groq API key.
- Embeddings use a HuggingFace model (internet required). If you prefer a cloud embedding API, replace `HuggingFaceEmbeddings`.
- Chromadb in-memory is fine for single-run extraction. For production, use a persistent vector DB like Pinecone or a persisted chroma directory.
""")