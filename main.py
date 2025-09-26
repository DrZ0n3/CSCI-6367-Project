import zipfile
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import tkinter as tk
from tkinter import ttk

# Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Collect documents
docs = []

# Path to zip
zip_path = "C:/Users/lizet/OneDrive/Personal/College/Graduate/Fall 2025/CSCI 6373/CSCI-6367-Project/Jan.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    html_files = [f for f in z.namelist() if f.endswith(".html")]

    for html_file in html_files:
        with z.open(html_file) as f:
            content = f.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
            docs.append(" ".join(words))  # store as a clean string

# docs are loaded, encode them
doc_embeddings = model.encode(docs, convert_to_numpy=True)

# Build FAISS index
d = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings)

def search_button_clicked():
    # Take user query
    query_text = search_entry.get()
    query_words = re.findall(r"\b[a-zA-Z]+\b", query_text.lower())
    query_clean = " ".join(query_words)

    # Embed query
    query_vector = model.encode([query_clean], convert_to_numpy=True)

    # Search top-k
    k = 2
    distances, indices = index.search(query_vector, k)

    # Clear previous results
    results_text.delete(1.0, tk.END)

    # Show results
    for i, idx in enumerate(indices[0]):
        result_snippet = docs[idx][:300] + "..."
        results_text.insert(tk.END, f"---Results {i + 1} ---\n{result_snippet}\n\n")  

# GUI
# Create the main window
root = tk.Tk()
root.title("Python Search Engine")

# Create widgets
search_label = ttk.Label(root, text="What do you want to search:")
search_label.pack(pady=5)

search_entry = ttk.Entry(root,width=50)
search_entry.pack(pady=5)

search_button = ttk.Button(root, text="Search", command=search_button_clicked)
search_button.pack(pady=5)

results_text = tk.Text(root, height=15, width=60)
results_text.pack(pady=10)

root.mainloop()

