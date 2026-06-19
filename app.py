import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import os
from numpy import dot
from numpy.linalg import norm

# ===== Setup (runs once, cached) =====

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    df = pd.read_csv("products.csv")
    df['chunk'] = df.apply(
        lambda row: f"Product: {row['name']}. Category: {row['category']}. Description: {row['description']}. SKU: {row['sku']}.",
        axis=1
    )
    return df

@st.cache_resource
def get_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

model = load_model()
df = load_data()
client = get_gemini_client()
product_embeddings = model.encode(df['chunk'].tolist())


# ===== RAG functions =====

def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def rag_chatbot(query, top_n=3):
    query_embedding = model.encode(query)
    similarities = []
    for i, product_emb in enumerate(product_embeddings):
        score = cosine_similarity(query_embedding, product_emb)
        similarities.append((i, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_n]

    context_parts = []
    for idx, score in top_results:
        product = df.iloc[idx]
        context_parts.append(
            f"- {product['name']} (SKU: {product['sku']}, Category: {product['category']}): {product['description']}"
        )
    context = "\n".join(context_parts)

    prompt = f"""You are a helpful assistant for a product catalog system.
Answer the user's question using ONLY the product information provided below.
If the answer isn't in the provided products, say you don't have that information.

Respond in a friendly, conversational way, in 2-3 sentences.
Explain WHY the product fits their need, don't just state the product name.

Available products:
{context}

User question: {query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


# ===== Streamlit UI =====

st.title("🛍️ MDM Product Catalog Assistant")
st.caption("Ask me anything about our product catalog!")

# Keep chat history across interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box
user_input = st.chat_input("Ask about our products...")

if user_input:
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get and show the bot's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chatbot(user_input)
            st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})