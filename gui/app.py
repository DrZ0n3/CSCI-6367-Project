import numpy as np
from backend.search import phrasal_search, perform_reformed_search
from backend.query_gobbledygook import booleanMagic
import tkinter as tk
from tkinter import ttk
import webbrowser
import spacy
import threading
from sklearn.neighbors import NearestNeighbors
import uuid

print("GUI function started")

nlp = spacy.load("en_core_web_sm")


used_tags = set()

def unique_tag():
    """Generate a globally unique short hash for Tkinter tags."""
    t = uuid.uuid4().hex[:10]
    while t in used_tags:
        t = uuid.uuid4().hex[:10]
    used_tags.add(t)
    return t


# INSERT LINK FUNCTION 
def insert_link(text_widget, label, tag, color="blue", centered=False):
    """
    Inserts a clickable hyperlink with a UNIQUE tag name.
    """
    start = text_widget.index(tk.INSERT)
    text_widget.insert(tk.END, label)
    end = text_widget.index(tk.INSERT)

    text_widget.tag_add(tag, start, end)

    cfg = {"foreground": color, "underline": True}
    if centered:
        cfg["justify"] = "center"

    text_widget.tag_config(tag, **cfg)

    text_widget.tag_bind(tag, "<Button-1>", lambda e, link=label: webbrowser.open_new(link))
    text_widget.tag_bind(tag, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
    text_widget.tag_bind(tag, "<Leave>", lambda e: text_widget.config(cursor=""))


# GUI MAIN FUNCTION
def search_engine_gui(inverted_index, doc_metadata, docs, doc_vectors, vocab):

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
        threading.Thread(target=perform_search).start()

    def perform_search():
        query_text = search_entry.get().strip()
        results_text.delete(1.0, tk.END)

        if not query_text:
            results_text.insert(tk.END, "Empty Input!\n")
            return

        # PHRASE SEARCH - Purple
        if query_text.startswith('"') and query_text.endswith('"'):
            phrase = query_text.strip('"')
            result_ids = phrasal_search(phrase, inverted_index)

            if result_ids:
                results_text.insert(tk.END, f"Phrase Match Found for \"{phrase}\":\n")

                for doc_id in sorted(result_ids):
                    metadata = doc_metadata[doc_id]
                    fname = metadata["filename"]
                    snippet = docs[doc_id][:300] + "..."

                    tag = unique_tag()

                    results_text.insert(tk.END, "\n")
                    insert_link(results_text, fname, tag, "purple", centered=True)

                    # URLs
                    results_text.insert(tk.END, "\n\nURLs Referenced: ")
                    urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3]]

                    for url in urls:
                        url_tag = unique_tag()
                        insert_link(results_text, url, url_tag, "purple")
                        results_text.insert(tk.END, ", ")

                    results_text.insert(tk.END, "\n" + snippet + "\n")

            else:
                results_text.insert(tk.END, f"No phrase match found for \"{phrase}\".\n")
            return

        # BOOLEAN SEARCH - BLACK
        BOOLEAN_OPS = {"and", "or", "not", "but", "&&", "||", "!", "&", "|"}

        tokens = query_text.lower().split()
        if any(tok in BOOLEAN_OPS for tok in tokens):
            result_ids = booleanMagic(query_text, inverted_index)

            if result_ids:
                results_text.insert(tk.END, "Boolean Match Found In:\n")

                for doc_id in result_ids:
                    metadata = doc_metadata[doc_id]
                    fname = metadata["filename"]
                    snippet = docs[doc_id][:300] + "..."

                    tag = unique_tag()
                    results_text.insert(tk.END, "\n")
                    insert_link(results_text, fname, tag, "black", centered=True)

                    # URLs
                    results_text.insert(tk.END, "\n\nURLs Referenced: ")
                    urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3]]

                    for url in urls:
                        url_tag = unique_tag()
                        insert_link(results_text, url, url_tag, "black")
                        results_text.insert(tk.END, ", ")

                    results_text.insert(tk.END, "\n" + snippet + "\n")

            else:
                results_text.insert(tk.END, "No Boolean matches found.\n")
            return

        # VECTOR SEARCH
        nn = NearestNeighbors(n_neighbors=5, metric="cosine")
        nn.fit(doc_vectors)
        search = perform_reformed_search(query_text, docs, nn, vocab, inverted_index)
        orig = search["initial_results"]
        reform = search["reformatted_results"]

        results_text.insert(tk.END, f"\nReformatted Query: {search['reformed_query']}\n")

        # ORIGINAL RESULTS — GREEN
        results_text.insert(tk.END, "\nTop Vector Space Results (Original):\n")

        for idx in orig:
            metadata = doc_metadata[idx]
            fname = metadata["filename"]
            snippet = docs[idx][:300] + "..."

            tag = unique_tag()

            results_text.insert(tk.END, "\n")
            insert_link(results_text, fname, tag, "green", centered=True)

            # URLs
            results_text.insert(tk.END, "\n\nURLs Referenced: ")
            urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3]]

            for url in urls:
                url_tag = unique_tag()
                insert_link(results_text, url, url_tag, "green")
                results_text.insert(tk.END, ", ")

            results_text.insert(tk.END, "\n" + snippet + "\n")

        results_text.insert(tk.END, "\n======================================================\n")

        # REFORMATTED RESULTS — BLUE
        results_text.insert(tk.END, "\nTop Vector Space Results (Reformatted):\n")

        for idx in reform:
            metadata = doc_metadata[idx]
            fname = metadata["filename"]
            snippet = docs[idx][:300] + "..."

            tag = unique_tag()

            results_text.insert(tk.END, "\n")
            insert_link(results_text, fname, tag, "blue", centered=True)

            # URLs
            results_text.insert(tk.END, "\n\nURLs Referenced: ")
            urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3]]

            for url in urls:
                url_tag = unique_tag()
                insert_link(results_text, url, url_tag, "blue")
                results_text.insert(tk.END, ", ")

            results_text.insert(tk.END, "\n" + snippet + "\n")

        results_text.insert(tk.END, "\n======================================================\n")


    # SHOW INVERTED INDEX SAMPLE
    def show_inverted_index_sample():
        results_text.delete(1.0, tk.END)

        for i, (word, data) in enumerate(list(inverted_index.items())[:5]):
            results_text.insert(tk.END, f"Word: '{word}' | DF: {data['df']}\n")
            for posting in data["postings"][:2]:
                results_text.insert(
                    tk.END,
                    f"  Doc ID: {posting['doc_id']}, TF: {posting['tf']}, "
                    f"TF-IDF: {posting.get('tf-idf', 0.0):.3f}, "
                    f"Positions: {posting['positions'][:3]}\n"
                )
            results_text.insert(tk.END, "-" * 40 + "\n")

    # Buttons
    ttk.Button(root, text="Search", command=search_button_clicked).pack(pady=5)
    ttk.Button(root, text="Show Inverted Index", command=show_inverted_index_sample).pack(pady=5)

    results_text = tk.Text(root, height=50, width=100)
    results_text.pack(pady=10)

    root.mainloop()
