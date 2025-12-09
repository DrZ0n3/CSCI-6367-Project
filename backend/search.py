import numpy as np
import spacy
import math
from collections import defaultdict, Counter
import re
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
# Embedding model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
# Lightweight spaCy pipeline
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

print("Search Engine Function Started")

# === PART ONE: NEAREST NEIGHBORS ===
def build_nearest_neighbors(doc_vectors):
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(doc_vectors)
    return nn


# === PART TWO: COSINE SIMILARITY ===
def cosine_similarity(query_vec, doc_vecs):
    qn = np.linalg.norm(query_vec)
    qn = qn if qn != 0 else 1e-10

    dn = np.linalg.norm(doc_vecs, axis=1)
    dn[dn == 0] = 1e-10

    return (doc_vecs @ query_vec) / (dn * qn)


# === PART THREE: QUERY VECTOR ===
def compute_query_vector(query_tokens, vocab, inverted_index, N):

    V = len(vocab)
    qvec = np.zeros(V, dtype=float)
    term_counts = Counter(query_tokens)

    logN = math.log(N)
    vocab_index = {word: i for i, word in enumerate(vocab)}
 # alias for faster lookup

    for token, tf in term_counts.items():
        if token in inverted_index:
            df = inverted_index[token]["df"]
            idf = logN - math.log(df + 1e-10)
            qvec[vocab_index[token]] = tf * idf

    return qvec



# === PART FOUR: PHRASAL SEARCH ===
def phrasal_search(phrase, inverted_index):

    # Tokenize only once
    phrase_tokens = [
        tok.lemma_.lower()
        for tok in nlp(phrase)
        if tok.is_alpha and not tok.is_stop
    ]

    if not phrase_tokens:
        return set()

    # If any word missing
    for t in phrase_tokens:
        if t not in inverted_index:
            return set()

    # Document intersection
    doc_sets = [
        {p["doc_id"] for p in inverted_index[t]["postings"]}
        for t in phrase_tokens
    ]
    candidate_docs = set.intersection(*doc_sets)

    # Check adjacency
    results = set()
    first_term = phrase_tokens[0]

    for doc_id in candidate_docs:

        # Collect all position lists
        pos_lists = []
        for t in phrase_tokens:
            for post in inverted_index[t]["postings"]:
                if post["doc_id"] == doc_id:
                    pos_lists.append(post["positions"])
                    break

        base_positions = pos_lists[0]

        # Sliding window adjacency check
        for pos in base_positions:
            if all((pos + i) in pos_lists[i] for i in range(1, len(pos_lists))):
                results.add(doc_id)
                break

    return results


# Stopwords
stop_words = set([
    "the","a","an","and","or","to","of","in","on","at","for","with",
    "is","it","this","that","as","are","be","by","from","was","were",
    "have","has","had","their","its","into","about","but"
])



# NEIGHBOR SEARCH
def search_docs(nn, query_vec, n_results=10):
    _, indices = nn.kneighbors([query_vec], n_neighbors=n_results)
    return indices[0]



# UNIQUE TERM COUNT
def select_top_pages(S, docs, top_k):
    scores = []
    for doc_id in S:
        score = len(set(docs[doc_id].split()))
        scores.append((score, doc_id))

    scores.sort(reverse=True)
    return [doc for _, doc in scores[:top_k]]



# keyword extraction
def extract_keywords(A, docs):
    words = []
    for doc_id in A:
        text = re.sub(r"[^a-z0-9 ]", " ", docs[doc_id].lower())
        for tok in text.split():
            if len(tok) > 2 and tok not in stop_words:
                words.append(tok)
    return Counter(words)



# correlation scoring
def corr_keywords_hybrid(query_text, freq_counter, select_n=10, alpha=0.75, beta=0.25):

    keywords = list(freq_counter.keys())

    # Semantic
    query_vec = semantic_model.encode([query_text])
    kw_vecs = semantic_model.encode(keywords)
    sims = cos(query_vec, kw_vecs)[0]

    sem_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # Frequency
    freqs = np.array([freq_counter[w] for w in keywords], dtype=float)
    freq_norm = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-8)

    # Combined score
    score = alpha * sem_norm + beta * freq_norm
    top_idx = np.argsort(score)[::-1][:select_n]

    return [keywords[i] for i in top_idx]


def reform_query(query_text, correlated):
    return " ".join(dict.fromkeys(query_text.split() + correlated))


# MAIN REFORMED SEARCH
def perform_reformed_search(query_text, docs, nn, vocab, inverted_index,
                            top_k=5, k=10, n_results=5):

    # Pre-tokenize query
    query_tokens = query_text.lower().split()

    # Query vector
    qvec = compute_query_vector(
        query_tokens, vocab, inverted_index, len(docs)
    )

    # First retrieval
    S = search_docs(nn, qvec, n_results)

    # Pick top pages
    A = select_top_pages(S, docs, top_k)

    # Extract keywords
    freq_counter = extract_keywords(A, docs)

    # Semantic + frequency correlation
    correlated = corr_keywords_hybrid(query_text, freq_counter, k)

    # Reform query
    rq = reform_query(query_text, correlated)

    # Second vector
    rq_vec = compute_query_vector(
        rq.lower().split(), vocab, inverted_index, len(docs)
    )

    # Second retrieval
    S_prime = search_docs(nn, rq_vec, n_results)

    # Merge
    final = list(dict.fromkeys(list(S) + list(S_prime)))

    return {
        "initial_results": list(S),
        "reformatted_results": list(S_prime),
        "top_docs": A,
        "keywords": list(freq_counter.keys()),
        "correlated_keywords": correlated,
        "reformed_query": rq,
        "final_results": final
    }
#test_search_engine()
#  #jarvis, tell these kids to get their shit together
