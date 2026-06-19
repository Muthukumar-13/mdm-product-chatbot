# MDM Product Catalog Assistant (RAG Chatbot)

## Problem
Data stewards and business users often need to search product catalogs
using natural language instead of SQL or manual spreadsheet searches.
This chatbot lets users ask plain English questions and get accurate,
grounded answers from the actual product catalog.

## What is RAG?
Retrieval-Augmented Generation combines semantic search with an LLM:
1. Retrieve relevant data from your own knowledge base
2. Pass that data + the user's question to an LLM
3. LLM generates an answer using ONLY that retrieved data

This prevents hallucination — the model can't make up answers using
its general training knowledge; it's grounded in your actual catalog.

## Pipeline
1. Load product catalog from CSV
2. Convert each product into a structured text chunk
3. Generate embeddings using sentence-transformers (runs locally, free)
4. Build semantic search using cosine similarity
5. Retrieve top matching products for any user question
6. Pass retrieved context + question to Gemini API with engineered prompt
7. Generate a natural, conversational, grounded answer
8. Wrap in a Streamlit chat interface with persistent history

## Key Results
- Semantic search correctly matches questions to products even when
  no keywords overlap (e.g. "bake a cake" correctly matched to a
  stand mixer)
- Model correctly declines to answer when information isn't available
  (tested with a pricing question — no pricing data exists in the
  catalog, and the model said so instead of guessing)
- Retry logic handles temporary API outages gracefully

## Libraries Used
- pandas - data manipulation
- sentence-transformers - local embedding generation
- google-genai - Gemini API for response generation
- streamlit - chat interface
- python-dotenv - secure API key management

## How to Run
1. Clone this repository
2. Create a virtual environment and activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file with your `GEMINI_API_KEY`
5. Run: `streamlit run app.py`

## Domain Context
Built on Master Data Management (MDM) domain knowledge - this prototype
demonstrates how natural language interfaces can sit on top of
structured master data, reducing reliance on manual queries.
