# query array example: A and B or C => ['a,b', 'c']
def query_array_encoder(query_string: str):
    query_string = query_string.replace("(", " ( ").replace(")", " ) ")
    tokens = query_string.split()
    precedence = {"or": 1, "and": 2}

    output_stack = []  # holds operands
    op_stack = []      # holds operators and '('

    def apply_operator():
        op = op_stack.pop()
        right = output_stack.pop()
        left = output_stack.pop()
        if op == "and":
            # join each combination with comma
            new_stack = [l + ',' + r for l in left for r in right]
        elif op == "or":
            # flatten union of lists
            new_stack = left + right
        else:
            raise ValueError(f"Unknown operator {op}")
        output_stack.append(new_stack)

    for token in tokens:
        token_lower = token.lower()
        if token == "(":
            op_stack.append(token)
        elif token == ")":
            while op_stack and op_stack[-1] != "(":
                apply_operator()
            op_stack.pop()  # remove '('
        elif token_lower in ("and", "or"):
            while op_stack and op_stack[-1] != "(" and precedence[op_stack[-1]] >= precedence[token_lower]:
                apply_operator()
            op_stack.append(token_lower)
        else:
            output_stack.append([token_lower])  

    # Apply remaining operators
    while op_stack:
        apply_operator()

    return output_stack[0]




def boolean_query(inverted_index, query_array):
    relevant_docs = set()  # still use set for intermediate operations

    for expression in query_array:
        keys = expression.split(',')  
        if not keys:
            continue

        docs_set = None  
        for key in keys:
            key_lower = key.lower()
            if key_lower in inverted_index:
                key_docs = set(post['doc_id'] for post in inverted_index[key_lower]['postings'])
                if docs_set is None:
                    docs_set = key_docs 
                else:
                    docs_set &= key_docs  
            else:
                docs_set = set()  
                break

        if docs_set:
            relevant_docs |= docs_set  

    return list(relevant_docs)  # convert set to list

# Sample inverted index for testing
inverted_index = {
    "a": {"postings": [{"doc_id": 1}, {"doc_id": 2}]},
    "b": {"postings": [{"doc_id": 2}, {"doc_id": 3}]},
    "c": {"postings": [{"doc_id": 1}, {"doc_id": 3}]},
    "d": {"postings": [{"doc_id": 3}]},
}

# Test queries and expected outputs for demonstration
test_queries = [
    ("A AND B", {2}),           # A,B => doc 2
    ("A OR B", {1, 2, 3}),      # A or B => docs 1,2,3
    ("A AND B OR C", {1,2,3}),  # ['A,B','C'] => union of docs
    ("A AND (B OR C)", {1,2,3}),# ['A,B','A,C'] => union
    ("B AND D", {3}),           # ['B,D'] => doc 3
    ("C OR D", {1,3}),          # ['C','D'] => docs 1,3
]

def run_boolean_query_tests():
    for i, (query, expected) in enumerate(test_queries, 1):
        encoded_query = query_array_encoder(query)
        result = boolean_query(inverted_index, encoded_query)
        print(f"Test {i}: {query}")
        print(f"Encoded query array: {encoded_query}")
        print(f"Relevant docs: {result}")
        print(f"Expected docs: {expected}")
        print(f"Test passed: {result == expected}")
        print("-" * 50)

# Run the tests
#run_boolean_query_tests()
