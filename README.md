# Weather-Extractor
RAG Weather Extractor — Scrape, Clean, Embed, Retrieve, Parse


## 🌦️ RAG Weather Extractor

A Streamlit app that scrapes weather forecast pages, cleans and processes the data, embeds it in a vector database, and uses a cloud LLM (Groq) to extract structured weather information.

This app uses a Retrieval-Augmented Generation (RAG) pipeline to convert messy webpage data into a neat JSON forecast table.

⸻

Features
	•	Scrape any weather website using:
	•	Requests (fast and simple)
	•	Selenium (optional, for JavaScript-heavy pages)
	•	Clean HTML pages by removing ads, scripts, headers, footers, and other noise.
	•	Chunk the text into smaller sections for easier understanding by the AI model.
	•	Generate embeddings with HuggingFace embeddings and store them in Chroma vector database (in-memory).
	•	Use Groq LLM to extract weather forecasts in structured JSON.
	•	Show both the parsed JSON output and raw LLM output for debugging.
	•	Configurable chunk size, overlap, and max chunks for fine-tuning.
	•	In-memory storage by default; optionally, can be extended to persistent vector stores.

⸻

Forecast Schema

The extracted forecast JSON has the following structure (Pydantic schema):

[
  {
    "day": "Today",
    "max_temp": "31°C",
    "min_temp": "23°C",
    "condition": "Sunny",
    "humidity": "40%",
    "precipitation": "4%",
    "wind_speed": "NW 10 km/h",
    "wind_direction": "NW"
  }
]


⸻

Installation
	1.	Clone the repository

git clone https://github.com/sancharika/Weather-Extractor.git
cd Weather-Extractor

	2.	Create a virtual environment

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

	3.	Install dependencies

pip install -r requirements.txt

Dependencies included:
	•	streamlit
	•	requests
	•	bs4 (BeautifulSoup)
	•	selenium (optional)
	•	selenium-stealth (optional)
	•	chromadb
	•	langchain
	•	langchain_groq
	•	pydantic
	•	huggingface-hub
	•	sentence-transformers

⸻

Configuration
	•	Groq API Key: Required to run the LLM.
	•	Groq Model: Default: moonshotai/kimi-k2-instruct-0905.
	•	Environment Variable Option (recommended for security):

export DEFAULT_GROQ_API_KEY="your_api_key_here"

In Streamlit, you can set secrets under Settings → Secrets:

DEFAULT_GROQ_API_KEY="your_api_key_here"


⸻

Usage
	1.	Run the app locally:

streamlit run app.py

	2.	Input the following in the UI:

	•	Weather URL: The page to scrape (e.g., TimeAndDate extended forecast).
	•	Groq API Key: Your API key (or leave blank if using environment variable).
	•	Groq Model: Model name for Groq.
	•	Use Selenium: Optional checkbox if the website needs JavaScript rendering.
	•	Chunk size, overlap, max chunks: Optional advanced settings to fine-tune text splitting.

	3.	Click “Extract forecast (RAG)”.

	•	Parsed JSON forecast will appear.
	•	Raw LLM output is shown for debugging.

⸻

How it Works
	1.	Scraping:
	•	Tries to fetch HTML via requests.
	•	Falls back to Selenium if enabled.
	2.	Cleaning HTML:
	•	Removes scripts, styles, nav, footer, ads, banners.
	3.	Chunking text:
	•	Breaks cleaned text into smaller pieces for embeddings.
	4.	Embeddings & VectorStore:
	•	Converts chunks into embeddings using HuggingFace.
	•	Stores embeddings in Chroma (in-memory).
	5.	RAG with Groq LLM:
	•	Retrieves relevant chunks from vector store.
	•	Sends them to Groq LLM with instructions to output strict JSON.
	6.	Display:
	•	Shows structured JSON forecast.
	•	Optionally shows raw LLM output for debugging.

⸻

Notes
	•	Selenium requires chromedriver installed in PATH.
	•	The pipeline is in-memory by default; you can extend it to persistent storage.
	•	Works best for pages with text-based weather data. Pages that load content via heavy JavaScript may require Selenium.
	•	Groq API usage is rate-limited; make sure your key is valid.

⸻

Example

Input URL:
https://www.timeanddate.com/weather/india/delhi/ext

Output JSON:

[
  {
    "day": "Today",
    "max_temp": "31°C",
    "min_temp": "23°C",
    "condition": "Sunny",
    "humidity": "40%",
    "precipitation": "4%",
    "wind_speed": "NW 10 km/h",
    "wind_direction": "NW"
  },
  {
    "day": "Tomorrow",
    "max_temp": "32°C",
    "min_temp": "24°C",
    "condition": "Partly cloudy",
    "humidity": "42%",
    "precipitation": "5%",
    "wind_speed": "NW 12 km/h",
    "wind_direction": "NW"
  }
]


⸻
