# --- Imports ---
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

# NLP imports
from spacy.lang.en import English
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# --- Setup NLP tools ---
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))

# Load spaCy model (disable parser + NER for speed)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

print("Indexer Function Begins")


# ---------------------------------------------------
#                 BUILD INDEX FUNCTION
# ---------------------------------------------------
def build_index(html_files):
    """
    Reads HTML files, extracts text, tokenizes/lemmatizes, builds:
      - Inverted index with TF, positions, TF-IDF
      - Document metadata
      - Document-term matrix
      - Vocabulary list
    """

    docs = []                       # Stores tokenized text for each doc
    html_filenames = []             # Stores filenames in same order
    doc_metadata = {}               # Stores metadata for each doc (tokens, hyperlinks, etc.)
    inverted_index = defaultdict(lambda: {"df": 0, "postings": []})

    TOKEN_RE = re.compile(r"[a-zA-Z]+")  # Only alphabetic tokens

    #                PROCESS EACH HTML FILE
    for html_file in html_files:

        # Load HTML file raw text
        with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        # Extract visible text only
        text = BeautifulSoup(html, "lxml", parse_only=SoupStrainer(text=True)).get_text()

        # Skip empty documents
        if not text.strip():
            continue

        # Prevent loading extremely large documents
        if len(text) > 900_000:
            print(f"Skipping long document ({len(text)} chars): {html_file}")
            continue

        #                  TOKENIZATION PIPELINE
        raw_tokens = TOKEN_RE.findall(text.lower())  # lowercase + regex filter
        tokens = [
            lemmatizer.lemmatize(t)
            for t in raw_tokens
            if t not in stop_words
        ]

        # Add processed doc to list
        docs.append(" ".join(tokens))
        doc_id = len(docs) - 1
        html_filenames.append(html_file)

        #                EXTRACT HYPERLINKS
        link_soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("a"))
        hyperlinks = []

        for a_tag in link_soup:
            if not isinstance(a_tag, Tag):
                continue  # skip comments/text nodes

            url = a_tag.get("href")
            if not url:
                continue

            hyperlinks.append({
                "url": url.strip(),
                "anchor_text": a_tag.get_text(strip=True),
                "visited": False
            })

        # Store metadata for this document
        doc_metadata[doc_id] = {
            "filename": html_file,
            "length": len(tokens),
            "tokens": tokens,
            "hyperlinks": hyperlinks
        }

        #               BUILD INVERTED INDEX
        position_map = {}
        for pos, token in enumerate(tokens):
            # Build list of positions where token appears in this doc
            position_map.setdefault(token, []).append(pos)

        # Add postings to global inverted index
        for token, positions in position_map.items():
            inverted_index[token]["df"] += 1      # document frequency
            inverted_index[token]["postings"].append({
                "doc_id": doc_id,
                "tf": len(positions),             # term frequency
                "positions": positions            # all positions in doc
            })

    #                  COMPUTE TF-IDF
    N = len(docs)  # total number of documents
    log = math.log

    for token, entry in inverted_index.items():
        idf = log(N / (entry["df"] + 1e-10))  # idf with epsilon for stability

        for p in entry["postings"]:
            p["tf-idf"] = p["tf"] * idf       # weight = tf × idf

    #              BUILD DOCUMENT-TERM MATRIX
    vocab = list(inverted_index.keys())
    word_to_index = {w: i for i, w in enumerate(vocab)}

    # Matrix shape: (num_docs x vocab_size)
    doc_vectors = np.zeros((N, len(vocab)))

    for token, entry in inverted_index.items():
        idx = word_to_index[token]  # column index

        for p in entry["postings"]:
            doc_vectors[p["doc_id"], idx] = p["tf-idf"]

    # Return everything needed for a working search engine
    return inverted_index, doc_metadata, docs, doc_vectors, vocab
