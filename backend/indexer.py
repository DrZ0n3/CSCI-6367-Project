
from bs4 import BeautifulSoup, Comment
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from collections import defaultdict
import math
import webbrowser

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

print("🟢Indexer Function Begins")

def build_index(html_files):
        # Global var
        docs = []
        html_filenames = []
        doc_metadata = {}
        inverted_index = defaultdict(lambda: {
            "df": 0,
            "postings": []
        })

        # === PART ONE: LOAD ZIP AND BUILD INVERTED INDEX ===
        for html_file in html_files:
                with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    soup = BeautifulSoup(content, "html.parser")

                    paragraphs = soup.find_all("p")
                    if paragraphs:
                        text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
                    else:
                        text = soup.get_text(separator=" ", strip=True)  # fallback if no <p> tags

                    
                    MAX_TEXT_LEN = 900_000  # spaCy limit is 1,000,000

                    if len(text) > MAX_TEXT_LEN:
                        print(f"Skipping long document ({len(text)} chars): {html_file}")
                        continue  
                    
                   
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
                    doc_id = len(docs)
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
        
        return inverted_index, doc_metadata, docs, doc_vectors, vocab
