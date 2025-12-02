
from bs4 import BeautifulSoup, Comment, SoupStrainer, Tag

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




def build_index(html_files):
    docs = []
    html_filenames = []
    doc_metadata = {}
    inverted_index = defaultdict(lambda: {"df": 0, "postings": []})

    TOKEN_RE = re.compile(r"[a-zA-Z]+")

    for html_file in html_files:
        with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        # Extract text 
        text = BeautifulSoup(html, "lxml", parse_only=SoupStrainer(text=True)).get_text()

        if not text.strip():
            continue

        if len(text) > 900_000:
            print(f"Skipping long document ({len(text)} chars): {html_file}")
            continue

        # --- tokenization ---
        raw_tokens = TOKEN_RE.findall(text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in raw_tokens if t not in stop_words]

        docs.append(" ".join(tokens))
        doc_id = len(docs) - 1
        html_filenames.append(html_file)

        # --- hyperlinks ---
        link_soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("a"))
        hyperlinks = []
        for a_tag in link_soup:
            if not isinstance(a_tag, Tag):
                continue  # skip Doctype, strings, comments, etc.

            url = a_tag.get("href")
            if not url:
                continue

            hyperlinks.append({
                "url": url.strip(),
                "anchor_text": a_tag.get_text(strip=True),
                "visited": False
            })


        doc_metadata[doc_id] = {
            "filename": html_file,
            "length": len(tokens),
            "tokens": tokens,
            "hyperlinks": hyperlinks
        }

        # --- inverted index ---
        position_map = {}
        for pos, token in enumerate(tokens):
            if token in position_map:
                position_map[token].append(pos)
            else:
                position_map[token] = [pos]

        for token, positions in position_map.items():
            inverted_index[token]["df"] += 1
            inverted_index[token]["postings"].append({
                "doc_id": doc_id,
                "tf": len(positions),
                "positions": positions
            })

    # --- TF-IDF ---
    N = len(docs)
    log = math.log

    for token, entry in inverted_index.items():
        idf = log(N / (entry["df"] + 1e-10))
        for p in entry["postings"]:
            p["tf-idf"] = p["tf"] * idf

    # --- Build vocab + vectors ---
    vocab = list(inverted_index.keys())
    word_to_index = {w: i for i, w in enumerate(vocab)}

    doc_vectors = np.zeros((N, len(vocab)))
    for token, entry in inverted_index.items():
        idx = word_to_index[token]
        for p in entry["postings"]:
            doc_vectors[p["doc_id"], idx] = p["tf-idf"]

    return inverted_index, doc_metadata, docs, doc_vectors, vocab
