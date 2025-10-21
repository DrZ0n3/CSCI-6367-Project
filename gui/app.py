from backend.search import phrasal_search, compute_query_vector, cosine_similarity
from backend.query_gobbledygook import booleanMagic
import tkinter as tk
from tkinter import ttk
import webbrowser
import spacy

print("🟢 GUI function started")


nlp = spacy.load("en_core_web_sm")

# === PART ONE: GUI FUNCTIONALITY ===
def open_link(url):
    webbrowser.open_new_tab(url)

# === PART TWO: LAUNCH GUI ===
def search_engine_gui( inverted_index, doc_metadata, docs, doc_vectors, vocab):
    global search_entry, results_text
    root = tk.Tk()
    root.title("MySearchEngine.com")

    search_label = ttk.Label(root, text="Enter Boolean or Free-text Query:")
    search_label.pack(pady=5)

    search_entry = ttk.Entry(root, width=50)
    search_entry.pack(pady=5)

    def search_button_clicked():
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
                    url = metadata["hyperlinks"][0]["url"] if metadata["hyperlinks"] else None

                    results_text.insert(tk.END, f"\n--- {fname} ---\n")
                    if url:
                        results_text.insert(tk.END, f"URL: {url}\n")
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
        result_ids = booleanMagic(" ".join(query_tokens),inverted_index)
        
        if result_ids is None:
            # Vector search fallback
            query_vec = compute_query_vector(query_tokens, vocab, inverted_index)
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

    results_text = tk.Text(root, height=15, width=70)
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