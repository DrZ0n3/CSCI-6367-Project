import numpy as np
import spacy
import math
from collections import defaultdict, Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

print("Search Engine Function Started")

def build_nearest_neighbors(doc_vectors):
    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(doc_vectors)
    return nn

# === PART THREE: COSINE SIMILARITY ===
def cosine_similarity(query_vec, doc_vecs):
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    query_norm = query_norm if query_norm != 0 else 1e-10
    doc_norms[doc_norms == 0] = 1e-10
    sims = doc_vecs @ query_vec / (doc_norms * query_norm)
    return sims

# === PART THREE: QUERY VECTOR ===
def compute_query_vector(query_tokens,vocab,word_to_index,inverted_index,N):
    query_vector = np.zeros(len(vocab))
    term_counts = defaultdict(int)
    for token in query_tokens:
        if token in word_to_index:
            term_counts[token] += 1
    for token, tf in term_counts.items():
        if token in inverted_index:
            df = inverted_index[token]["df"]
            idf = math.log(N / (df + 1e-10))
            tfidf = tf * idf
            index = word_to_index[token]
            query_vector[index] = tfidf
    return query_vector


# === PART FOUR: PHRASAL SEARCH ===
def phrasal_search(phrase,inverted_index):

    # Tokenize phrase
    phrase_tokens = [
        token.lemma_.lower()
        for token in nlp(phrase)
        if token.is_alpha and not token.is_stop
    ]

    if not phrase_tokens:
        return set()

    # If a word is missing, return nothing.
    for token in phrase_tokens:
        if token not in inverted_index:
            return set()

    # Find the documents
    candidate_docs = [
        {p["doc_id"] for p in inverted_index[token]["postings"]}
        for token in phrase_tokens
    ]
    common_docs = set.intersection(*candidate_docs)

    # Check adjacency
    matching_docs = set()
    for doc_id in common_docs:
        # Collect position lists for each term
        position_lists = []
        for token in phrase_tokens:
            for posting in inverted_index[token]["postings"]:
                if posting["doc_id"] == doc_id:
                    position_lists.append(posting["positions"])
                    break

        # Check if consecutive
        base_positions = position_lists[0]
        for pos in base_positions:
            if all((pos + i) in position_lists[i] for i in range(1, len(position_lists))):
                matching_docs.add(doc_id)
                break  # one match is enough for the doc

    return matching_docs


stop_words = set(["the","a","an","and","or","to","of","in","on","at","for","with"])


def build_nn_index(doc_vectors):
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(doc_vectors)
    return nn


def search_docs(nn, query_vec, n_results=10):
    distances, indices = nn.kneighbors([query_vec], n_neighbors=n_results)
    return indices[0]   # return document IDs


def select_top_pages(S, docs, top_k):
    scores = []
    for doc_id in S:
        text = docs[doc_id].lower()
        unique_terms = set(text.split())
        score = len(unique_terms)
        scores.append((score, doc_id))

    scores.sort(reverse=True)
    return [doc_id for _, doc_id in scores[:top_k]]


def extract_keywords(A, docs, top_n=20):
    words = []
    for doc_id in A:
        text = docs[doc_id].lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        tokens = text.split()

        for tok in tokens:
            if tok not in stop_words and len(tok) > 2:
                words.append(tok)

    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


def corr_keywords(query_text, K, select_n):
    q_terms = set(query_text.lower().split())
    correlations = []

    for kw in K:
        score = 0
        for q in q_terms:
            if q in kw or kw in q:
                score += 2
            elif abs(len(q) - len(kw)) <= 2:
                score += 1
        correlations.append((score, kw))

    correlations.sort(reverse=True)
    return [kw for score, kw in correlations[:select_n] if score > 0]


def reform_query(query_text, correlated):
    return (query_text + " " + " ".join(correlated)).strip()


def perform_reformed_search(query_text, docs, doc_vectors, vocab, inverted_index, top_k=5, select_n=5, n_results=10):

    query_tokens = query_text.lower().split()
    query_vec = compute_query_vector(query_tokens, vocab, inverted_index, len(docs))

    nn = build_nn_index(doc_vectors)
    #some documentation: learn to do this please
    # 1) First retrieval S
    S = search_docs(nn, query_vec, n_results=n_results)

    # 2) Select A ⊂ S
    A = select_top_pages(S, docs, top_k)

    # 3) Extract keyword list K from A
    K = extract_keywords(A, docs)

    # 4) Correlate keywords with query
    correlated = corr_keywords(query_text, K, select_n)

    # 5) Reformulate query
    reformed_q = reform_query(query_text, correlated)

    # 6) Compute reformulated query vector
    rq_tokens = reformed_q.lower().split()
    rq_vec = compute_query_vector(rq_tokens, vocab, inverted_index, len(docs))

    # 9) Second retrieval S'
    S_prime = search_docs(nn, rq_vec, n_results=n_results)

    # 10) Final result = S ∪ S'
    final_results = list(dict.fromkeys(list(S) + list(S_prime)))

    return {
        "initial_results": S,
        "top_docs": A,
        "keywords": K,
        "correlated_keywords": correlated,
        "reformed_query": reformed_q,
        "final_results": final_results
    }
