import numpy as np
import spacy
import math
from collections import defaultdict

from sklearn.neighbors import NearestNeighbors

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger", "textcat"])

print("Search Engine Function Started")

# FAST: Precache log function
_log = math.log

def build_nearest_neighbors(doc_vectors):
    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(doc_vectors)
    return nn

# COSINE SIMILARITY 
def cosine_similarity(query_vec, doc_vecs):
    # Precompute norms once
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    doc_norms[doc_norms == 0] = 1e-10

    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        query_norm = 1e-10

    # Vectorized cosine similarity
    return (doc_vecs @ query_vec) / (doc_norms * query_norm)


# QUERY VECTOR (optimized without changing parameters)

def compute_query_vector(query_tokens, vocab, inverted_index, N):
    vocab_len = len(vocab)
    query_vector = np.zeros(vocab_len, dtype=float)

    # Build word→index map ONE TIME
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Count token frequencies
    term_counts = defaultdict(int)
    for token in query_tokens:
        idx = word_to_index.get(token)
        if idx is not None:
            term_counts[token] += 1

    # Compute tf-idf
    for token, tf in term_counts.items():
        entry = inverted_index.get(token)
        if entry is None:
            continue
        df = entry["df"]
        idf = _log(N / (df + 1e-10))
        query_vector[word_to_index[token]] = tf * idf

    return query_vector

# PHRASE SEARCH — major optimizations but same functionality
def phrasal_search(phrase, inverted_index):
    # Tokenize & normalize once
    phrase_tokens = [
        tok.lemma_.lower()
        for tok in nlp(phrase)
        if tok.is_alpha and not tok.is_stop
    ]

    if not phrase_tokens:
        return set()

    # If any token missing → no doc matches
    for t in phrase_tokens:
        if t not in inverted_index:
            return set()

    # Intersect all document sets quickly
    candidate_docs = None
    for token in phrase_tokens:
        token_docs = {p["doc_id"] for p in inverted_index[token]["postings"]}
        if candidate_docs is None:
            candidate_docs = token_docs
        else:
            candidate_docs &= token_docs

        # early exit
        if not candidate_docs:
            return set()

    matching_docs = set()
    tok_count = len(phrase_tokens)

    for doc_id in candidate_docs:
        # gather position lists for each token in this doc
        position_lists = []
        for token in phrase_tokens:
            for post in inverted_index[token]["postings"]:
                if post["doc_id"] == doc_id:
                    position_lists.append(post["positions"])
                    break

        base_positions = position_lists[0]

        # fast adjacency check
        for pos in base_positions:
            ok = True
            for i in range(1, tok_count):
                if (pos + i) not in position_lists[i]:
                    ok = False
                    break
            if ok:
                matching_docs.add(doc_id)
                break

    return matching_docs
