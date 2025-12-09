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

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

print("Indexer Function Begins")

# Precompile regex once (huge speedup)
TOKEN_RE = re.compile(r'\b[a-zA-Z]+\b')

def tokenize_text_fast(text):
    tokens = TOKEN_RE.findall(text.lower())
    sw = stop_words
    lemma = lemmatizer.lemmatize
    return [lemma(t) for t in tokens if t not in sw]


def build_index(html_files):

    # Pre-bind locals (speed boost)
    tokenize = tokenize_text_fast
    BeautifulSoup_local = BeautifulSoup
    inverted_index = defaultdict(lambda: {"df": 0, "postings": []})

    docs = []
    html_filenames = []
    doc_metadata = {}

    MAX_TEXT_LEN = 900_000

    # === PART ONE: PARSE HTML + EXTRACT TOKENS ===
    for html_file in html_files:

        # HTML reading
        with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        # Use lxml paser
        soup = BeautifulSoup_local(html, "lxml")

        # get_text() is expensive → call once
        text = soup.get_text()
        if not text.strip():
            continue

        if len(text) > MAX_TEXT_LEN:
            print(f"Skipping long document ({len(text)} chars): {html_file}")
            continue

        # TOKEN PIPELINE
        tokens = tokenize(text)
        clean_text = " ".join(tokens)

        #  extraction of hyperlinks
        hyperlinks = []
        append_hyper = hyperlinks.append
        for a in soup.find_all("a", href=True):
            append_hyper({
                "url": a["href"].strip(),
                "anchor_text": a.get_text(strip=True) or None,
                "visited": False
            })

        # store metadata
        doc_id = len(docs)
        docs.append(clean_text)
        html_filenames.append(html_file)

        doc_metadata[doc_id] = {
            "filename": html_file,
            "length": len(tokens),
            "tokens": tokens,
            "hyperlinks": hyperlinks
        }

        # Build position map
        position_map = defaultdict(list)
        pm_add = position_map
        for pos, token in enumerate(tokens):
            pm_add[token].append(pos)

        # Add postings to inverted index
        for token, positions in position_map.items():
            entry = inverted_index[token]
            entry["df"] += 1
            entry["postings"].append({
                "doc_id": doc_id,
                "tf": len(positions),
                "positions": positions
            })


    # === PART TWO: COMPUTE TF-IDF ===
    N = len(docs)
    log = math.log

    for token, entry in inverted_index.items():
        idf = log(N / (entry["df"] + 1e-10))
        for p in entry["postings"]:
            p["tf-idf"] = p["tf"] * idf


    # === PART THREE: BUILD VOCAB + DOC-VECTORS ===
    vocab = list(inverted_index.keys())
    word_to_index = {w: i for i, w in enumerate(vocab)}

    doc_vectors = np.zeros((len(doc_metadata), len(vocab)))

    for token, entry in inverted_index.items():
        col = word_to_index[token]
        for p in entry["postings"]:
            doc_vectors[p["doc_id"], col] = p["tf-idf"]

    return inverted_index, doc_metadata, docs, doc_vectors, vocab
