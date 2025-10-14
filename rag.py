# -----------------------
#  Main RAG pipeline
# -----------------------
import json, re
from typing import List

# Selenium imports (used only if user toggles it on or scraping fails)
from scraper import scrape_with_requests, scrape_with_selenium

#preprocess imports
from preprocess import clean_html, chunk_text

# LangChain + Groq + Vector DB imports
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# NOTE: Replace this import if you use a different wrapper for Groq LLM
from langchain_groq import ChatGroq



# -----------------------
#  Forecast schema (Pydantic)
# -----------------------
class ForecastItem(BaseModel):
    day: str
    max_temp: str | None = None
    min_temp: str | None = None
    condition: str | None = None
    humidity: str | None = None
    precipitation: str | None = None
    wind_speed: str | None = None
    wind_direction: str | None = None

# Parser to enforce JSON return
parser = PydanticOutputParser(pydantic_object=ForecastItem)


def build_embeddings_model(model_name="all-MiniLM-L6-v2"):
    # This uses a Hugging Face embeddings model - internet required at runtime.
    return HuggingFaceEmbeddings(model_name=model_name)

def build_vectorstore(text_chunks: List[str], embeddings) -> Chroma:
    # In-memory Chroma; no persistence by default (persist_directory=None)
    chroma = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return chroma

def build_groq_llm(api_key: str, model_name: str, temperature: float = 0.1):
    # Thin wrapper around Groq Chat model used in your original code
    return ChatGroq(api_key=api_key, model=model_name, temperature=temperature)

def run_rag_extraction(
    url: str,
    groq_api_key: str,
    groq_model: str,
    use_selenium: bool = False,
    max_chunks: int = 8
):
    # 1) Scrape (try requests, fallback to selenium if chosen)
    html = ""
    try:
        html = scrape_with_requests(url)
    except Exception as e:
        if use_selenium:
            html = scrape_with_selenium(url)
        else:
            # If requests failed and selenium not allowed, try selenium once
            try:
                html = scrape_with_selenium(url)
            except Exception as ex:
                raise RuntimeError(f"Both requests and selenium scraping failed: {e} / {ex}")

    # 2) Clean and chunk
    cleaned = clean_html(html)
    if not cleaned or len(cleaned) < 50:
        raise RuntimeError("Page cleaned to very little text — scraping may have failed.")
    chunks = chunk_text(cleaned)
    # limit number of chunks to reduce cost
    chunks = chunks[: max(1, max_chunks)]

    # 3) Embeddings + vectorstore
    embeddings = build_embeddings_model()
    vectordb = build_vectorstore(chunks, embeddings)

    # 4) Retriever + Groq LLM
    llm = build_groq_llm(api_key=groq_api_key, model_name=groq_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 5) Compose system prompt to instruct LLM to output strict JSON list
    format_instructions = parser.get_format_instructions()
    user_prompt = f"""
You are given chunks of cleaned webpage text (retrieved from a weather forecast page).
Extract the weather forecast items and return a JSON array of forecast objects that match this Pydantic schema:

{format_instructions}

Return ONLY valid JSON (a list of objects). If fields are unknown put null. Try to extract: day (e.g., Today, Mon, 2025-10-15), max_temp, min_temp, condition, humidity, precipitation, wind_speed, wind_direction.
Use concise values, e.g., '31°C', '4%', 'NW 10 km/h'.
"""

    # Run the RAG chain
    result = qa_chain.run(user_prompt)
    # Attempt to parse as JSON array — sometimes the LLM may return one object per line
    try:
        parsed = json.loads(result)
        return parsed, result
    except Exception:
        # Try to extract JSON substring
        m = re.search(r"(\[.*\])", result, re.S)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return parsed, result
            except Exception:
                pass
        # If still fails, return raw text as fallback
        return None, result