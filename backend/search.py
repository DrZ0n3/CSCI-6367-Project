import numpy as np
import spacy
import math
from collections import defaultdict, Counter
import re
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
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
def compute_query_vector(query_tokens,vocab,inverted_index,N):
    
    query_vector = np.zeros(len(vocab))
    term_counts = defaultdict(int)
    word_to_index = {word: i for i, word in enumerate(vocab)}
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

stop_words = set([
    "the","a","an","and","or","to","of","in","on","at","for","with",
    "is","it","this","that","as","are","be","by","from","was","were",
    "have","has","had","their","its","into","about","but"
])

def build_nn_index(doc_vectors):
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(doc_vectors)
    return nn


def search_docs(nn, query_vec, n_results=10):
    distances, indices = nn.kneighbors([query_vec], n_neighbors=n_results)
    return indices[0]


def select_top_pages(S, docs, top_k):
    scores = []
    for doc_id in S:
        text = docs[doc_id].lower()
        unique_terms = set(text.split())
        score = len(unique_terms)
        scores.append((score, doc_id))

    scores.sort(reverse=True)
    return [doc_id for _, doc_id in scores[:top_k]]


def extract_keywords(A, docs, top_n=50):
    words = []
    for doc_id in A:
        text = docs[doc_id].lower()
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        tokens = text.split()

        for tok in tokens:
            if tok not in stop_words and len(tok) > 2:
                words.append(tok)

    return Counter(words)



def corr_keywords_hybrid(query_text, freq_counter, select_n=10, alpha=0.75, beta=0.25):

    keywords = list(freq_counter.keys())

    # --- semantic similarity ---
    query_vec = semantic_model.encode([query_text])
    kw_vecs = semantic_model.encode(keywords)

    sims = cos(query_vec, kw_vecs)[0]
    sem_norm = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # --- frequency weighting ---
    freqs = np.array([freq_counter[w] for w in keywords], dtype=float)
    freq_norm = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-8)

    # --- hybrid score ---
    final_score = alpha * sem_norm + beta * freq_norm

    # rank
    ranked_ids = np.argsort(final_score)[::-1]
    top = [keywords[i] for i in ranked_ids[:select_n]]

    return top



def reform_query(query_text, correlated):
    query_words = query_text.split()
    all_words = list(dict.fromkeys(query_words + correlated))

    return " ".join(all_words)


def perform_reformed_search(query_text, docs, doc_vectors, vocab, inverted_index, 
                            top_k=5, k=10, n_results=5):

    query_tokens = query_text.lower().split()
    query_vec = compute_query_vector(
        query_tokens,
        vocab,
        inverted_index,
        len(docs)
)
    nn = build_nn_index(doc_vectors)

    # 1) First retrieval S
    S = search_docs(nn, query_vec, n_results=n_results)

    # 2) Select A ⊂ S
    A = select_top_pages(S, docs, top_k)

    # 3) Extract keyword frequencies
    freq_counter = extract_keywords(A, docs)

    # 4) Hybrid semantic + statistical correlation
    correlated = corr_keywords_hybrid(query_text, freq_counter, k)

    # 5) Reformulate query
    reformed_q = reform_query(query_text, correlated)

    # 6) Compute reformulated query vector
    rq_tokens = reformed_q.lower().split()
    rq_vec = compute_query_vector(rq_tokens,vocab,inverted_index,len(docs)
    )

    # 7) Second retrieval S'
    S_prime = search_docs(nn, rq_vec, n_results=n_results)

    # 8) Final result = S ∪ S'
    final_results = list(dict.fromkeys(list(S) + list(S_prime)))

    return {
        "initial_results": list(S),
        "reformatted_results": list(S_prime),
        "top_docs": A,
        "keywords": list(freq_counter.keys()),
        "correlated_keywords": correlated,
        "reformed_query": reformed_q,
        "final_results": final_results
    }

#hey jarvis build me a test function
def test_search_engine():
    print("\n=== RUNNING TEST ===")

    # ---------------------------
    # 1) Toy document collection
    # ---------------------------
    docs = {
        0: "The quick brown fox jumps over the lazy dog.",
        1: "A fast brown fox leaps across a sleepy canine.",
        2: "Deep learning methods require large datasets.",
        3: "Neural networks are used for deep learning.",
        4: "The dog is lazy but the fox is quick."
    }

    # ---------------------------
    # 2) Build vocabulary + index
    # ---------------------------
    vocab = {}
    word_to_index = {}
    inverted_index = defaultdict(lambda: {"df": 0, "postings": []})
    index_counter = 0

    for doc_id, text in docs.items():
        tokens = [t.lemma_.lower() for t in nlp(text)]
        position = 0
        seen = set()

        for tok in tokens:
            if tok.isalpha() and tok not in stop_words:

                # add to vocab
                if tok not in vocab:
                    vocab[tok] = index_counter
                    word_to_index[tok] = index_counter
                    index_counter += 1

                # update inverted index
                idx = vocab[tok]
                if tok not in seen:
                    inverted_index[tok]["df"] += 1
                    seen.add(tok)

                inverted_index[tok]["postings"].append({
                    "doc_id": doc_id,
                    "positions": [position]
                })
            position += 1

    # ---------------------------
    # 3) Build doc vectors (TF-IDF)
    # ---------------------------
    N = len(docs)
    V = len(vocab)
    doc_vectors = np.zeros((N, V))

    for doc_id, text in docs.items():
        tokens = [t.lemma_.lower() for t in nlp(text)]
        counts = Counter([tok for tok in tokens if tok in vocab])

        for tok, tf in counts.items():
            df = inverted_index[tok]["df"]
            idf = math.log(N / (df + 1e-10))
            doc_vectors[doc_id, vocab[tok]] = tf * idf

    # ---------------------------
    # 4) Test query
    # ---------------------------
    query = "brown fox"

    results = perform_reformed_search(
        query_text=query,
        docs=docs,
        doc_vectors=doc_vectors,
        vocab=vocab,
        inverted_index=inverted_index,
        top_k=3,
        select_n=3,
        n_results=5
    )

    # ---------------------------
    # 5) Print results
    # ---------------------------
    print("\n=== RESULTS ===")
    print("Initial Results:", results["initial_results"])
    print("Top Docs:", results["top_docs"])
    print("Extracted Keywords:", results["keywords"])
    print("Correlated Keywords:", results["correlated_keywords"])
    print("Reformed Query:", results["reformed_query"])
    print("Final Results:", results["final_results"])

    print("\n=== TEST COMPLETE ===\n")
#test_search_engine()
#jarvis, tell these kids to get their shit together

