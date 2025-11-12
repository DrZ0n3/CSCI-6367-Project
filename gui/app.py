from backend.search import phrasal_search, compute_query_vector, cosine_similarity
from backend.query_gobbledygook import booleanMagic
import tkinter as tk
from tkinter import ttk
import webbrowser
import spacy
import threading
from sklearn.neighbors import NearestNeighbors

print("GUI function started")


nlp = spacy.load("en_core_web_sm")


# === PART ONE: GUI FUNCTIONALITY ===
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

        


# === PART TWO: LAUNCH GUI ===
def search_engine_gui( inverted_index, doc_metadata, docs, doc_vectors, vocab):
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

        # Detect phrase search (quoted text)
        if query_text.startswith('"') and query_text.endswith('"'):
            phrase = query_text.strip('"')
            result_ids = phrasal_search(phrase, inverted_index)
            if result_ids:
                results_text.insert(tk.END, f"Phrase Match Found for \"{phrase}\":\n")
                for doc_id in sorted(result_ids):
                    metadata = doc_metadata[doc_id]
                    fname = metadata["filename"]
                    snippet = docs[doc_id][:300] + "..."
                    results_text.insert(tk.END, "\n")
                    insert_link(results_text,fname, doc_id, True)
                    results_text.insert(tk.END, "\n\n")
                    results_text.insert(tk.END, "URLs Referenced: ")

                    urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3] if "url" in link]
                    for i, url in enumerate(urls):
                        insert_link(results_text, url, f"url_{i}")
                        if i < len(urls) - 1:  
                            results_text.insert(tk.END, ", ")
                    results_text.insert(tk.END, "\n")        
                    results_text.insert(tk.END, f"{snippet}\n")
            else:
                results_text.insert(tk.END, f"No phrase match found for \"{phrase}\".\n")
            return

        query_tokens = [
            token.lemma_.lower()
            for token in nlp(query_text)
            if token.is_alpha and not token.is_stop
        ]

        # Check if boolean queries exist.
        result_ids = booleanMagic(query_text,inverted_index)
        
        if result_ids is None:
            # Vector search fallback
            N = len(docs)
            query_vec = compute_query_vector(query_tokens, vocab, inverted_index, N)
            distances, top_indices = nn.kneighbors([query_vec])
            results_text.insert(tk.END, "\nTop Vector Space Results:\n")
            for rank, idx in enumerate(top_indices[0]):
                score = 1 - distances[0][rank]  # convert cosine distance -> similarity
                fname = doc_metadata[idx]["filename"]
                snippet = docs[idx][:300] + "..."
                results_text.insert(tk.END, f"\n--- Rank {rank + 1}: ")
                insert_link(results_text, fname, idx)
        elif result_ids:

            results_text.insert(tk.END, "Boolean Match Found In:\n")
            for doc_id in result_ids:
                metadata = doc_metadata[doc_id]
                fname = metadata["filename"]
                snippet = docs[doc_id][:300] + "..."
                url = metadata["hyperlinks"][0]["url"] if metadata["hyperlinks"] else None

                results_text.insert(tk.END, "\n")
                insert_link(results_text,fname, doc_id, True)
                results_text.insert(tk.END, "\n\n")
                results_text.insert(tk.END, "URLs Referenced: ")

                urls = [link["url"] for link in metadata.get("hyperlinks", [])[:3] if "url" in link]
                for i, url in enumerate(urls):
                    insert_link(results_text, url, f"url_{i}")
                    if i < len(urls) - 1:  
                        results_text.insert(tk.END, ", ")
                results_text.insert(tk.END, "\n")
                results_text.insert(tk.END, f"{snippet}\n")
        else:
            results_text.insert(tk.END, "No match found.\n")

    def show_inverted_index_sample():
        results_text.delete(1.0, tk.END)

        sample_size = 5
        for i, (word, data) in enumerate(inverted_index.items()):
            if i >= sample_size:
                break
            results_text.insert(tk.END, f"Word: '{word}' | DF: {data['df']}\n")
            for posting in data["postings"][:2]:  # show first 2 postings only
                results_text.insert(
                    tk.END,
                    f"  Doc ID: {posting['doc_id']}, TF: {posting['tf']}, "
                    f"TF-IDF: {posting.get('tf-idf', 0.0):.3f}, "
                    f"Positions: {posting['positions'][:3]}\n"
                )
            results_text.insert(tk.END, "-" * 40 + "\n")


    search_button = ttk.Button(root, text="Search", command=search_button_clicked)
    search_button.pack(pady=5)

    search_button = ttk.Button(root, text="Show Inverted Index", command=show_inverted_index_sample)
    search_button.pack(pady=5)

    results_text = tk.Text(root, height=50, width=100)
    results_text.pack(pady=10)

    root.mainloop()



# def test_index_and_metadata():
#     print("\n=== Document Metadata Sample ===")
#     for doc_id, metadata in list(doc_metadata.items())[:3]:
#         print(f"Doc ID: {doc_id}")
#         print(f"Filename: {metadata['filename']}")
#         print(f"Length (tokens): {metadata['length']}")
#         print(f"Hyperlinks: {[link['url'] for link in metadata['hyperlinks']][:3]}")
#         print("Sample tokens:", metadata["tokens"][:5])
#         print("-" * 40)