import numpy as np
from backend.search import phrasal_search, compute_query_vector, cosine_similarity
from backend.query_gobbledygook import booleanMagic
import tkinter as tk
from tkinter import ttk
import webbrowser
import spacy
import threading
from sklearn.neighbors import NearestNeighbors
import threading

print("GUI function started")

# Load minimal spaCy model – MUCH faster
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "textcat"])

# Predefined Boolean operators (do not recreate each search)
BOOLEAN_OPERATORS = {"and", "or", "not", "but", "&&", "||", "!", "&", "|"}


def insert_link(text_widget, url, doc_id, centered=False):
    start_idx = text_widget.index(tk.INSERT)
    text_widget.insert(tk.END, url)
    end_idx = text_widget.index(tk.INSERT)

    text_widget.tag_add(doc_id, f"{start_idx}", end_idx.strip())
    if centered:
        text_widget.tag_config(doc_id, foreground="blue", underline=True, justify="center")
    else:
        text_widget.tag_config(doc_id, foreground="blue", underline=True)

    text_widget.tag_bind(doc_id, "<Button-1>", lambda e, link=url: webbrowser.open_new(link))
    text_widget.tag_bind(doc_id, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
    text_widget.tag_bind(doc_id, "<Leave>", lambda e: text_widget.config(cursor=""))


def search_engine_gui(inverted_index, doc_metadata, docs, doc_vectors, vocab):

    # Precompute normalized vectors ONCE (big speed boost)
    doc_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_norms[doc_norms == 0] = 1e-10
    normalized_docs = doc_vectors / doc_norms

    # Precompute vocab lookup ONCE
    word_to_index = {word: i for i, word in enumerate(vocab)}

    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(doc_vectors)

    global search_entry, results_text

    root = tk.Tk()
    root.title("MySearchEngine.com")

    search_label = ttk.Label(root, text="Enter Boolean or Free-text Query:")
    search_label.pack(pady=5)

    search_entry = ttk.Entry(root, width=50)
    search_entry.pack(pady=5)

    def search_button_clicked():
        threading.Thread(target=perform_search, daemon=True).start()

    def perform_search():
        query_text = search_entry.get().strip()
        results_text.delete(1.0, tk.END)

        if not query_text:
            results_text.insert(tk.END, "Empty Input!\n")
            return

        # -------- PHRASE SEARCH --------
        if query_text.startswith('"') and query_text.endswith('"'):
            phrase = query_text[1:-1]
            result_ids = phrasal_search(phrase, inverted_index)

            if not result_ids:
                results_text.insert(tk.END, f"No phrase match found for \"{phrase}\".\n")
                return

            results_text.insert(tk.END, f"Phrase Match Found for \"{phrase}\":\n")
            for doc_id in sorted(result_ids):
                metadata = doc_metadata[doc_id]
                fname = metadata["filename"]
                snippet = docs[doc_id][:300] + "..."

                results_text.insert(tk.END, "\n")
                insert_link(results_text, fname, doc_id, True)
                results_text.insert(tk.END, "\n\nURLs Referenced: ")

                urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3]]
                for i, url in enumerate(urls):
                    insert_link(results_text, url, f"url_{i}")
                    if i < len(urls) - 1:
                        results_text.insert(tk.END, ", ")

                results_text.insert(tk.END, "\n" + snippet + "\n")
            return

        # -------- TOKENIZE QUERY ONCE--------
        query_tokens = [
            tok.lemma_.lower()
            for tok in nlp(query_text)
            if tok.is_alpha and not tok.is_stop
        ]

        # -------- BOOLEAN DETECTION--------
        bool_query = any(tok in BOOLEAN_OPERATORS for tok in query_text.lower().split())

        if bool_query:
            result_ids = booleanMagic(query_text, inverted_index)

            results_text.insert(tk.END, "Boolean Match Found In:\n")
            for doc_id in result_ids:
                metadata = doc_metadata[doc_id]
                fname = metadata["filename"]
                snippet = docs[doc_id][:300] + "..."

                results_text.insert(tk.END, "\n")
                insert_link(results_text, fname, doc_id, True)
                results_text.insert(tk.END, "\n\nURLs Referenced: ")

                urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3]]
                for i, url in enumerate(urls):
                    insert_link(results_text, url, f"url_{i}")
                    if i < len(urls) - 1:
                        results_text.insert(tk.END, ", ")

                results_text.insert(tk.END, "\n" + snippet + "\n")
            return

        # -------- VECTOR SEARCH--------
        N = len(docs)
        query_vec = compute_query_vector(query_tokens, vocab, inverted_index, N)

        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            results_text.insert(tk.END, "\nNo matching terms found.\n")
            return

        q = query_vec / qnorm
        scores = normalized_docs @ q

        k = 10
        top_indices = np.argsort(scores)[::-1][:k]

        results_text.insert(tk.END, "\nTop Vector Space Results:\n")

        for rank, idx in enumerate(top_indices):
            fname = doc_metadata[idx]["filename"]
            snippet = docs[idx][:300] + "..."

            results_text.insert(tk.END, f"\n--- Rank {rank + 1}: ")
            insert_link(results_text, fname, idx)
            results_text.insert(tk.END, f"\n{snippet}\n")

    # Buttons
    search_button = ttk.Button(root, text="Search", command=search_button_clicked)
    search_button.pack(pady=5)

    def show_inverted_index_sample():
        results_text.delete(1.0, tk.END)
        for i, (word, data) in enumerate(list(inverted_index.items())[:5]):
            results_text.insert(tk.END, f"Word: '{word}' | DF: {data['df']}\n")
            for posting in data["postings"][:2]:
                results_text.insert(
                    tk.END,
                    f"  Doc ID: {posting['doc_id']}, TF: {posting['tf']}, "
                    f"TF-IDF: {posting['tf-idf']:.3f}, "
                    f"Positions: {posting['positions'][:3]}\n"
                )
            results_text.insert(tk.END, "-" * 40 + "\n")

    index_button = ttk.Button(root, text="Show Inverted Index", command=show_inverted_index_sample)
    index_button.pack(pady=5)

    results_text = tk.Text(root, height=50, width=100)
    results_text.pack(pady=10)

    root.mainloop()
