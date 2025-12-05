import numpy as np
from scipy.sparse import csr_matrix
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser","ner","tagger","textcat"])

def compute_query_vector(query_tokens, vocab, inverted_index, N):
    word_to_index = {w: i for i, w in enumerate(vocab)}
    query_vec = np.zeros(len(vocab))
    term_counts = {}
    for token in query_tokens:
        if token in word_to_index:
            term_counts[token] = term_counts.get(token, 0) + 1

    logN = np.log
    for token, tf in term_counts.items():
        entry = inverted_index.get(token)
        if entry:
            df = entry["df"]
            idf = logN(N / (df + 1e-10))
            query_vec[word_to_index[token]] = tf * idf
    return query_vec

def cosine_similarity(query_vec, doc_vectors):
    """doc_vectors: sparse csr_matrix"""
    qnorm = np.linalg.norm(query_vec)
    if qnorm == 0:
        return np.zeros(doc_vectors.shape[0])
    # Sparse dot product
    scores = doc_vectors.dot(query_vec) / (qnorm * np.sqrt(doc_vectors.power(2).sum(axis=1)).A1)
    return scores

def phrasal_search(phrase, inverted_index):
    phrase_tokens = [tok.lemma_.lower() for tok in nlp(phrase) if tok.is_alpha and not tok.is_stop]
    if not phrase_tokens:
        return set()
    # Candidate docs: intersection
    candidate_docs = None
    for token in phrase_tokens:
        if token not in inverted_index:
            return set()
        token_docs = {p["doc_id"] for p in inverted_index[token]["postings"]}
        candidate_docs = token_docs if candidate_docs is None else candidate_docs & token_docs
        if not candidate_docs:
            return set()
    # Verify adjacency
    result = set()
    for doc_id in candidate_docs:
        pos_lists = []
        for token in phrase_tokens:
            for p in inverted_index[token]["postings"]:
                if p["doc_id"] == doc_id:
                    pos_lists.append(set(p["positions"]))
                    break
        base = pos_lists[0]
        for pos in base:
            if all((pos + i) in pos_lists[i] for i in range(1, len(pos_lists))):
                result.add(doc_id)
                break
    return result
