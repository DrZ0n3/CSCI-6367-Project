import spacy
nlp = spacy.load("en_core_web_sm")

def booleanMagic(query_string: str, inverted_index):
    query_string = query_string.replace("(", " ( ").replace(")", " ) ")
    tokens = query_string.split()
    precedence = {"or": 1, "and": 2, "but": 3, "not": 4}

    output_stack = []
    op_stack = []

    all_docs = set()
    for postings in inverted_index.values():
        all_docs.update(post['doc_id'] for post in postings['postings'])

    def apply_operator():
        op = op_stack.pop()

        if op == "not":
            operand = output_stack.pop()
            output_stack.append(all_docs - operand)
        else:
            right = output_stack.pop()
            left = output_stack.pop()
            if op == "but":
                output_stack.append(left - right)
            elif op == "and":
                output_stack.append(left & right)
            elif op == "or":
                output_stack.append(left | right)

    def normalize_text(text):
        """Normalize a token or phrase using spaCy (lemma + lowercase + remove stopwords/punct)."""
        doc = nlp(text)
        return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]

    for token in tokens:
        token_lower = token.lower()

        if token_lower == "(":
            op_stack.append(token_lower)

        elif token_lower == ")":
            while op_stack and op_stack[-1] != "(":
                apply_operator()
            op_stack.pop()

        elif token_lower in ("and", "or", "but"):
            while op_stack and op_stack[-1] != "(" and precedence[op_stack[-1]] >= precedence[token_lower]:
                apply_operator()
            op_stack.append(token_lower)

        elif token_lower == "not":
            op_stack.append(token_lower)

        else:
            # normalize the query term
            normalized_terms = normalize_text(token_lower)

            # union of all matching docs for all normalized forms of this token
            docs = set()
            for term in normalized_terms:
                if term in inverted_index:
                    docs.update(post['doc_id'] for post in inverted_index[term]['postings'])
            output_stack.append(docs)

    while op_stack:
        apply_operator()

    return list(output_stack.pop() if output_stack else [])


inverted_index = {
    "run": {"postings": [{"doc_id": 1}, {"doc_id": 2}]},
    "walk": {"postings": [{"doc_id": 2}, {"doc_id": 3}]},
    "fast": {"postings": [{"doc_id": 1}, {"doc_id": 3}]},
    "slow": {"postings": [{"doc_id": 3}]},
    "happy": {"postings": [{"doc_id": 2}]},
}
def run_tests_verbose():
    tests = [
        # (query, expected_result)
        ("run", {1, 2}),
        ("run and fast", {1}),
        ("run or slow", {1, 2, 3}),
        ("not run", {3}),
        ("walk but fast", {2}),
        ("run and (fast or slow)", {1}),  # fixed expectation
        ("running and fast", {1}),  # normalization check
        ("not walk and fast", {1}),
        ("(run and walk) or not happy", {1, 2, 3}),
    ]

    for i, (query, expected) in enumerate(tests, start=1):
        result = set(booleanMagic(query, inverted_index))
        if result == expected:
            print(f"✅ Test {i} passed: '{query}' → {result}")
        else:
            print(f"❌ Test {i} FAILED: '{query}'")
            print(f"   Expected: {expected}")
            print(f"   Got     : {result}")
            raise AssertionError(f"Test {i} failed")

    print("\nAll tests passed!")

# Run verbose tests
run_tests_verbose()