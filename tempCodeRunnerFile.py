# === Boolean Query Parser ===
# def tokenize_query(query):
#     return re.findall(r'\w+|AND|OR|NOT|\(|\)', query, re.IGNORECASE)

# def eval_query(query_tokens, doc_tokens):
#     doc_token_set = set(doc_tokens)

#     def word_in_doc(word):
#         return word.lower() in doc_token_set

#     expression = ""
#     for token in query_tokens:
#         if token.upper() == "AND":
#             expression += " and "
#         elif token.upper() == "OR":
#             expression += " or "
#         elif token.upper() == "NOT":
#             expression += " not "
#         elif token in ("(", ")"):
#             expression += token
#         else:
#             expression += f"word_in_doc('{token}')"

#     try:
#         return eval(expression)
#     except Exception as e:
#         print(f"Error evaluating query: {e}")
#         return False


