def booleanMagic(query_string: str, inverted_index):
    query_string = query_string.replace("(", " ( ").replace(")", " ) ")
    tokens = query_string.split()
    precedence = {"or": 1, "and": 2, "not": 3}

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
            if op == "and":
                output_stack.append(left & right)
            elif op == "or":
                output_stack.append(left | right)

    for token in tokens:
        token_lower = token.lower()

        if token_lower == "(":
            op_stack.append(token_lower)

        elif token_lower == ")":
            while op_stack and op_stack[-1] != "(":
                apply_operator()
            op_stack.pop()

        elif token_lower in ("and", "or"):
            while op_stack and op_stack[-1] != "(" and precedence[op_stack[-1]] >= precedence[token_lower]:
                apply_operator()
            op_stack.append(token_lower)

        elif token_lower == "not":
            op_stack.append(token_lower)

        else:
            if token_lower in inverted_index:
                docs = set(post['doc_id'] for post in inverted_index[token_lower]['postings'])
            else:
                docs = set()
            output_stack.append(docs)

    while op_stack:
        apply_operator()

    return list(output_stack.pop() if output_stack else [])

def test_booleanMagic():
    inverted_index = {
        "apple": {"postings": [{"doc_id": 1}, {"doc_id": 2}, {"doc_id": 4}]},
        "banana": {"postings": [{"doc_id": 2}, {"doc_id": 3}]},
        "cherry": {"postings": [{"doc_id": 3}, {"doc_id": 4}]},
        "date": {"postings": [{"doc_id": 1}]},
    }

    tests = [
        # basic queries
        ("apple and banana", {2}),
        ("apple or banana", {1, 2, 3, 4}),
        ("apple and not banana", {1, 4}),
        ("not apple", {3}),  # only doc 3 lacks 'apple'
        # parentheses and precedence
        ("apple and (banana or cherry)", {2, 4}),
        ("(apple or banana) and cherry", {3, 4}),
        ("not (banana or cherry)", {1}),
        ("(not banana) and (not cherry)", {1}),
    ]

    for query, expected in tests:
        result = set(booleanMagic(query, inverted_index))
        print(f"Query: {query}\n → Result: {result}, Expected: {expected}")
        assert result == expected, f"Test failed for query: {query}"

    print("\n✅ All tests passed!")

# ---- Run the tests ----
test_booleanMagic()
