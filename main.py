#Damian SID: 20489422
#Jose Luis Castellanos SID:20576044
#Pablo SID:20581962
#Lizeth Chavez SID:20523200

import zipfile
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import tkinter as tk
from tkinter import ttk

import spacy

from collections import defaultdict
import math

# Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Load english lang model 
nlp = spacy.load("en_core_web_sm")

# Collect documents
docs = []
html_filenames = []
doc_metadata = {}
inverted_index = defaultdict(lambda: {
    "df": 0,
    "postings": []
})

# Path to zip
zip_path = "C:/Users/lizet/OneDrive/Personal/College/Graduate/Fall 2025/CSCI 6373/CSCI-6367-Project/Jan.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    html_files = [f for f in z.namelist() if f.endswith(".html")]

    for doc_id, html_file in enumerate(html_files):
        with z.open(html_file) as f:
            content = f.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")

            #spaCy tokenizer
            text = soup.get_text(separator=" ", strip=True)
            doc = nlp(text)

            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_stop
            ]

            #Extract hyperlinks
            hyperlinks = []
            for a_tag in soup.find_all('a', href=True):
                url = a_tag['href']
                hyperlinks.append({"url": url, "visited": False})

            #Store document info
            clean_text = " ".join(tokens)
            docs.append(clean_text) 
            html_filenames.append(html_file)


            doc_metadata[doc_id] = {
            
                    "filename": html_file,
                    "length": len(tokens),
                    "tokens": tokens,
                    "hyperlinks": hyperlinks
            }

            #inverted index
            position_map = defaultdict(list)
            for pos, token in enumerate(tokens):
                position_map[token].append(pos)

            for token, position in position_map.items():
                tf = len(position)
                inverted_index[token]["df"] += 1
                inverted_index[token]["postings"].append({
                    "doc_id": doc_id,
                    "tf": tf,
                    "positions": position
                })

#compute tf-idf score
N = len(doc)
for token, entry in inverted_index.items():
    df = entry["df"]
    idf = math.log(N/df)
    for posting in entry["postings"]:
        posting["tf-idf"] = posting["tf"] * idf

# docs are loaded, encode them
doc_embeddings = model.encode(docs, convert_to_numpy=True)

# Build FAISS index
d = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(doc_embeddings)




def search_button_clicked():
    # Take user query
    query_text = search_entry.get()

    #If input is empty
    if not query_text:
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, "Empty Input!\n")
        return
    
    query_words = re.findall(r"\b[a-zA-Z]+\b", query_text.lower())

    # === SEMANTIC SEARCH (FAISS) ===
    query_clean = " ".join(query_words)

    # Embed query
    query_vector = model.encode([query_clean], convert_to_numpy=True)

    # Search top-k
    k = 2
    distances, indices = index.search(query_vector, k)

    # Clear previous results
    results_text.delete(1.0, tk.END)

    results_text.insert(tk.END, "\nTop semantic results: \n")

    # Show results
    for i, idx in enumerate(indices[0]):
        if idx < len(docs):
            result_snippet = docs[idx][:300] + "..."
            results_text.insert(tk.END, f"---Results {i + 1} ({html_filenames[idx]})---\n{result_snippet}\n") 

def test_index_and_metadata():
        print("\n=== Document Metadata Sample ===")
        for doc_id, metadata in list(doc_metadata.items())[:3]:  # print first 3 entries
            print(f"Doc ID: {doc_id}")
            print(f"Filename: {metadata['filename']}")
            print(f"Length (tokens): {metadata['length']}")
            print(f"Hyperlinks: {[link['url'] for link in metadata['hyperlinks']][:3]}")
            print("Sample tokens:", metadata["tokens"][:5])
            print("-" * 40)

        print("\n=== Inverted Index Sample ===")
        for word, data in list(inverted_index.items())[:5]:  # first 5 words
            print(f"Word: '{word}'")
            print(f"Document Frequency (df): {data['df']}")
            for posting in data["postings"][:2]:  # print first 2 postings
                doc_id = posting["doc_id"]
                tf = posting["tf"]
                tfidf = posting.get("tf-idf", 0.0)
                positions = posting["positions"][:5]  # show up to 5 positions
                print(f"  Doc ID: {doc_id}, TF: {tf}, TF-IDF: {tfidf:.2f}, Positions: {positions}")
            print("-" * 40) 

test_index_and_metadata()

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

