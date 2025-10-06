#Damian SID: 20489422
#Jose Luis Castellanos SID:20576044
#Pablo SID:20581962
#Lizeth Chavez SID:20523200

import query_gobbledygook
import zipfile
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer
import numpy as np

import tkinter as tk
from tkinter import ttk

import spacy

from collections import defaultdict
import math


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
zip_path = "./Jan.zip"

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

            clean_text = " ".join(tokens)

            #Extract hyperlinks
            hyperlinks = []
            for a_tag in soup.find_all('a', href=True):
                url = a_tag['href']
                hyperlinks.append({"url": url, "visited": False})

            #Store document info
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

# === Boolean Query Parser ===
def tokenize_query(query):
    return re.findall(r'\w+|AND|OR|NOT|\(|\)', query, re.IGNORECASE)

def eval_query(query_tokens, doc_tokens):
    doc_token_set = set(doc_tokens)

    def word_in_doc(word):
        return word.lower() in doc_token_set

    expression = ""
    for token in query_tokens:
        if token.upper() == "AND":
            expression += " and "
        elif token.upper() == "OR":
            expression += " or "
        elif token.upper() == "NOT":
            expression += " not "
        elif token in ("(", ")"):
            expression += token
        else:
            expression += f"word_in_doc('{token}')"

    try:
        return eval(expression)
    except Exception as e:
        print(f"Error evaluating query: {e}")
        return False




def search_button_clicked():
    # Take user query
    query_text = search_entry.get().strip()
    results_text.delete(1.0, tk.END)

    #If input is empty
    if not query_text:
        results_text.insert(tk.END, "Empty Input!\n")
        return
    
    query_tokens = tokenize_query(query_text)
    results = []

    # Clear previous results
    results_text.delete(1.0, tk.END)

    for doc_id, metadata in doc_metadata.items():
        if eval_query(query_tokens, metadata["tokens"]):
            results.append(metadata["filename"])

    if results:
        results_text.insert(tk.END, "Found match in:\n")
        for fname in results:
            doc_id = html_filenames.index(fname)
            snippet = docs[doc_id][:300] + "..."
            results_text.insert(tk.END, f"\n--- {fname} ---\n{snippet}\n")
    else:
        results_text.insert(tk.END, "No match found.\n")
 

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
search_label = ttk.Label(root, text="Enter Boolean Query:")
search_label.pack(pady=5)

search_entry = ttk.Entry(root,width=50)
search_entry.pack(pady=5)

search_button = ttk.Button(root, text="Search", command=search_button_clicked)
search_button.pack(pady=5)

results_text = tk.Text(root, height=15, width=60)
results_text.pack(pady=10)

# print(query_gobbledygook.boolean_query(inverted_index, query_gobbledygook.query_array_encoder("cat and dog and rat")))

root.mainloop()

