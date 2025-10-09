# Damian SID: 20489422
# Jose Luis Castellanos SID:20576044
# Pablo SID:20581962
# Lizeth Chavez SID:20523200

import query_gobbledygook
import zipfile
from bs4 import BeautifulSoup, Comment
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import tkinter as tk
from tkinter import ttk
import spacy
from collections import defaultdict
import math
import webbrowser

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Global var
docs = []
html_filenames = []
doc_metadata = {}
inverted_index = defaultdict(lambda: {
    "df": 0,
    "postings": []
})

# === PART ONE: LOAD ZIP AND BUILD INVERTED INDEX ===
zip_path = "./Jan.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    html_files = [f for f in z.namelist() if f.endswith(".html")]

    for doc_id, html_file in enumerate(html_files):
        with z.open(html_file) as f:
            content = f.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")

            # for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            #    comment.extract()

            paragraphs = soup.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
            else:
                text = soup.get_text(separator=" ", strip=True)  # fallback if no <p> tags

            # Tokenize using spaCy
            doc = nlp(text)
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_stop
            ]
            clean_text = " ".join(tokens)

            # Hyperlinks
            hyperlinks = []
            for a_tag in soup.find_all('a', href=True):
                url = a_tag['href']
                hyperlinks.append({"url": url, "visited": False})

            # Store metadata
            docs.append(clean_text)
            html_filenames.append(html_file)
            doc_metadata[doc_id] = {
                "filename": html_file,
                "length": len(tokens),
                "tokens": tokens,
                "hyperlinks": hyperlinks
            }

            # Build inverted index
            position_map = defaultdict(list)
            for pos, token in enumerate(tokens):
                position_map[token].append(pos)

            for token, positions in position_map.items():
                tf = len(positions)
                inverted_index[token]["df"] += 1
                inverted_index[token]["postings"].append({
                    "doc_id": doc_id,
                    "tf": tf,
                    "positions": positions
                })

# === PART ONE: COMPUTE TF-IDF ===
N = len(docs)
for token, entry in inverted_index.items():
    df = entry["df"]
    idf = math.log(N / (df + 1e-10))  # prevent div by zero
    for posting in entry["postings"]:
        posting["tf-idf"] = posting["tf"] * idf

# === PART THREE: BUILD VOCAB AND DOC-VECTOR MATRIX ===
vocab = list(inverted_index.keys())
word_to_index = {word: idx for idx, word in enumerate(vocab)}

doc_vectors = np.zeros((len(doc_metadata), len(vocab)))
for token, entry in inverted_index.items():
    token_index = word_to_index[token]
    for posting in entry["postings"]:
        doc_id = posting["doc_id"]
        tfidf = posting["tf-idf"]
        doc_vectors[doc_id][token_index] = tfidf

# === PART THREE: COSINE SIMILARITY ===
def cosine_similarity(query_vec, doc_vecs):
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    query_norm = query_norm if query_norm != 0 else 1e-10
    doc_norms[doc_norms == 0] = 1e-10
    sims = doc_vecs @ query_vec / (doc_norms * query_norm)
    return sims

# === PART THREE: QUERY VECTOR ===
def compute_query_vector(query_tokens):
    query_vector = np.zeros(len(vocab))
    term_counts = defaultdict(int)
    for token in query_tokens:
        if token in word_to_index:
            term_counts[token] += 1
    for token, tf in term_counts.items():
        if token in inverted_index:
            df = inverted_index[token]["df"]
            idf = math.log(N / (df + 1e-10))
            tfidf = tf * idf
            index = word_to_index[token]
            query_vector[index] = tfidf
    return query_vector

# === PART THREE: QUERY SEARCHER (Q1–Q4) ===
def boolean_query(query_text):
    text = query_text.lower().strip()

    # Text has "but"
    if " but " in text:
        left, right = text.split(" but ", 1)
        left_terms = [t.lemma_.lower() for t in nlp(left) if t.is_alpha and not t.is_stop]
        right_terms = [t.lemma_.lower() for t in nlp(right) if t.is_alpha and not t.is_stop]

        left_docs = set()
        for term in left_terms:
            if term in inverted_index:
                left_docs |= {p["doc_id"] for p in inverted_index[term]["postings"]}

        right_docs = set()
        for term in right_terms:
            if term in inverted_index:
                right_docs |= {p["doc_id"] for p in inverted_index[term]["postings"]}

        return left_docs - right_docs

    # Text has "and"
    elif " and " in text:
        terms = [t.strip() for t in text.split(" and ") if t.strip()]
        doc_sets = []
        for term in terms:
            term_tokens = [t.lemma_.lower() for t in nlp(term) if t.is_alpha and not t.is_stop]
            docs = set()
            for token in term_tokens:
                if token in inverted_index:
                    docs |= {p["doc_id"] for p in inverted_index[token]["postings"]}
            if docs:
                doc_sets.append(docs)
        if not doc_sets:
            return set()
        return set.intersection(*doc_sets)

    # Text has "or"
    elif " or " in text:
        terms = [t.strip() for t in text.split(" or ") if t.strip()]
        result_docs = set()
        for term in terms:
            term_tokens = [t.lemma_.lower() for t in nlp(term) if t.is_alpha and not t.is_stop]
            for token in term_tokens:
                if token in inverted_index:
                    result_docs |= {p["doc_id"] for p in inverted_index[token]["postings"]}
        return result_docs

    # Nothing, so no boolean.
    return None


# === PART FOUR: PHRASAL SEARCH ===
def phrasal_search(phrase):

    # Tokenize phrase
    phrase_tokens = [
        token.lemma_.lower()
        for token in nlp(phrase)
        if token.is_alpha and not token.is_stop
    ]

    if not phrase_tokens:
        return set()

    # If a word is missing, return nothing.
    for token in phrase_tokens:
        if token not in inverted_index:
            return set()

    # Find the documents
    candidate_docs = [
        {p["doc_id"] for p in inverted_index[token]["postings"]}
        for token in phrase_tokens
    ]
    common_docs = set.intersection(*candidate_docs)

    # Check adjacency
    matching_docs = set()
    for doc_id in common_docs:
        # Collect position lists for each term
        position_lists = []
        for token in phrase_tokens:
            for posting in inverted_index[token]["postings"]:
                if posting["doc_id"] == doc_id:
                    position_lists.append(posting["positions"])
                    break

        # Check if consecutive
        base_positions = position_lists[0]
        for pos in base_positions:
            if all((pos + i) in position_lists[i] for i in range(1, len(position_lists))):
                matching_docs.add(doc_id)
                break  # one match is enough for the doc

    return matching_docs

# === PART ONE: GUI FUNCTIONALITY ===
def open_link(url):
    webbrowser.open_new_tab(url)


def search_button_clicked():
    query_text = search_entry.get().strip()
    results_text.delete(1.0, tk.END)

    if not query_text:
        results_text.insert(tk.END, "Empty Input!\n")
        return

    # Detect phrase search (quoted text)
    if query_text.startswith('"') and query_text.endswith('"'):
        phrase = query_text.strip('"')
        result_ids = phrasal_search(phrase)
        if result_ids:
            results_text.insert(tk.END, f"Phrase Match Found for \"{phrase}\":\n")
            for doc_id in sorted(result_ids):
                metadata = doc_metadata[doc_id]
                fname = metadata["filename"]
                snippet = docs[doc_id][:300] + "..."
                url = metadata["hyperlinks"][0]["url"] if metadata["hyperlinks"] else None

                results_text.insert(tk.END, f"\n--- {fname} ---\n")
                if url:
                    results_text.insert(tk.END, f"URL: {url}\n")
                results_text.insert(tk.END, f"{snippet}\n")
        else:
            results_text.insert(tk.END, f"No phrase match found for \"{phrase}\".\n")
        return

    # Detect vector search (no Boolean operators)
    # is_vector_query = all(op not in query_text.lower() for op in ["and", "or", "but"])

    query_tokens = [
        token.lemma_.lower()
        for token in nlp(query_text)
        if token.is_alpha and not token.is_stop
    ]

    # Check if boolean queries exist.
    result_ids = boolean_query(query_text)

    if result_ids is None:
        # Vector search fallback
        query_vec = compute_query_vector(query_tokens)
        similarities = cosine_similarity(query_vec, doc_vectors)
        top_indices = similarities.argsort()[::-1][:5]
        results_text.insert(tk.END, "\nTop Vector Space Results:\n")
        for rank, idx in enumerate(top_indices):
            if similarities[idx] > 0:
                fname = doc_metadata[idx]["filename"]
                snippet = docs[idx][:300] + "..."
                score = similarities[idx]
                results_text.insert(
                    tk.END, f"\n--- Rank {rank + 1}: {fname} (Score: {score:.3f}) ---\n{snippet}\n"
                )
    elif result_ids:

        results_text.insert(tk.END, "Boolean Match Found In:\n")
        for doc_id in result_ids:
            metadata = doc_metadata[doc_id]
            fname = metadata["filename"]
            snippet = docs[doc_id][:300] + "..."
            url = metadata["hyperlinks"][0]["url"] if metadata["hyperlinks"] else None

            results_text.insert(tk.END, f"\n--- {fname} ---\n")
            if url:
                results_text.insert(tk.END, f"URL: {url}\n")
            results_text.insert(tk.END, f"{snippet}\n")
    else:
        results_text.insert(tk.END, "No match found.\n")

# === PART TWO: LAUNCH GUI ===
def search_engine_gui():
    global search_entry, results_text
    root = tk.Tk()
    root.title("Python Search Engine")

    search_label = ttk.Label(root, text="Enter Boolean or Free-text Query:")
    search_label.pack(pady=5)

    search_entry = ttk.Entry(root, width=50)
    search_entry.pack(pady=5)

    search_button = ttk.Button(root, text="Search", command=search_button_clicked)
    search_button.pack(pady=5)

    results_text = tk.Text(root, height=15, width=70)
    results_text.pack(pady=10)

    root.mainloop()

# === TEST INDEX OUTPUT ===


def test_index_and_metadata():
    print("\n=== Document Metadata Sample ===")
    for doc_id, metadata in list(doc_metadata.items())[:3]:
        print(f"Doc ID: {doc_id}")
        print(f"Filename: {metadata['filename']}")
        print(f"Length (tokens): {metadata['length']}")
        print(f"Hyperlinks: {[link['url'] for link in metadata['hyperlinks']][:3]}")
        print("Sample tokens:", metadata["tokens"][:5])
        print("-" * 40)

    print("\n=== Inverted Index Sample ===")
    for word, data in list(inverted_index.items())[:5]:
        print(f"Word: '{word}' | DF: {data['df']}")
        for posting in data["postings"][:2]:
            print(f"  Doc ID: {posting['doc_id']}, TF: {posting['tf']}, TF-IDF: {posting.get('tf-idf', 0.0):.3f}, Positions: {posting['positions'][:3]}")
        print("-" * 40)


# === RUN ===
test_index_and_metadata()
search_engine_gui()
