import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the data
data = pd.read_csv("with_encoding.csv")

# Convert string embeddings to list of floats
data['encoding'] = data['encoding'].apply(lambda x: [float(num) for num in x.replace('[','').replace(']','').replace('\n','').split()])

# Build the Faiss index
index = faiss.IndexFlatL2(len(data['encoding'][0]))
for i, emb in enumerate(data['encoding']):
    index.add(np.array([emb], dtype=np.float32))

# Load sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Streamlit App
st.title("Facebook AI Similarity Search App")

# User input for query
query = st.text_input("Enter your query:")

if query:
    # Transform query sentence into embeddings
    query_embedding = model.encode(query)

    # Search in Faiss index
    k = 5  # Number of results to retrieve
    query_embedding = np.array([query_embedding], dtype=np.float32)
    _, result_indices = index.search(query_embedding, k)

    # Display results
    st.subheader("Top 5 Results:")
    for i, result_index in enumerate(result_indices[0]):
        st.write(f"{i + 1}. Index: {result_index}")
        st.write(f"   Paragraph: {data['charaka'][result_index]}")
        st.write(f"   Book: {data['name'][result_index]}")
        st.write("---")
