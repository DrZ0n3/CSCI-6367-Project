
from bs4 import BeautifulSoup, Comment
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from collections import defaultdict
import math
import webbrowser
import nltk
import pickle

import spacy
from spacy.lang.en import English
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

# Load spaCy language model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

print("Indexer Function Begins")

def tokenize_text_fast(text):
    # Lowercase and extract words only
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Remove stopwords + lemmatize
    return [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]

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
                    html = f.read()
                    soup = BeautifulSoup(html, "html.parser")


                    # paragraphs = soup.find_all("p")
                    # if paragraphs:
                    #     text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
                    # else:
                    #     text = soup.get_text(separator=" ", strip=True)  # fallback if no <p> tags

                    text = soup.get_text()
                    if not text.strip():
                        continue

                    MAX_TEXT_LEN = 900_000  # spaCy limit is 1,000,000

                    if len(text) > MAX_TEXT_LEN:
                        print(f"Skipping long document ({len(text)} chars): {html_file}")
                        continue  
                    
                   
                    tokens = tokenize_text_fast(text)

                    clean_text = " ".join(tokens)

                   # Hyperlinks with anchor text
                    hyperlinks = []
                    for a_tag in soup.find_all("a", href=True):
                        url = a_tag["href"].strip()
                        anchor_text = a_tag.get_text(separator=" ", strip=True) or None  # Extract anchor text
                        hyperlinks.append({
                            "url": url,
                            "anchor_text": anchor_text,
                            "visited": False
                        })


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
        
