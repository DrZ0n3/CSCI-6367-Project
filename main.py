#Damian SID: 20489422
#Jose Luis Castellanos SID:20576044
#Pablo SID:20581962
#Lizeth Chavez SID:20523200

import zipfile
from bs4 import BeautifulSoup
import re
from collections import Counter

# Using the Huggubg Face model
from sentence_transformers import SentenceTransformer
import faiss # facebook thing
import numpy as np

# loading embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Storing docuemnts in docs
docs = []

# Embed documents
doc_embeddings = model.encode(docs, convert_to_numpy=True)

# Building index using FAISS
d = doc_embeddings.shape[1]  # embedding dimension
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(doc_embeddings)

# Testing Query
query = "puppy playing outdoors"
query_vector = model.encode([query], convert_to_numpy=True)


# Displaying top 2 results 
k = 2
distances, indices = index.search(query_vector, k)

# Print results
for idx in indices[0]:
    print(docs[idx])


# Current path for jan.zip
zip_path = r""

print("Not Google")

with zipfile.ZipFile(zip_path, "r") as z:
    html_files = [f for f in z.namelist() if f.endswith(".html")]

    file_word_counts = {}  # dictionary: {filename: Counter(words)}

    for html_file in html_files:
        with z.open(html_file) as f:
            content = f.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(content, "html.parser")

            # Extract plain text
            text = soup.get_text(separator=" ", strip=True)

            # Keep only words (letters only)
            words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

            docs = docs(words)

            # NOT IMPORTANT

            # # Count word frequencies
            # counter = Counter(words)

            # file_word_counts[html_file] = counter

            # # Show top 5 words in this file
            # print(f"{html_file} -> {len(words)} total words")
            # for word, count in counter.most_common(5):
            #     print(f"   {word}: {count}")


print(docs)
query = input("What do you want to search: ")

query = re.findall(r"\b[a-zA-Z]+\b", query.lower())
print(query)                              

# for currFile, in file_word_counts: 
    
#     for word, count in counter.most_common(2):
#         print(f"TEST:  {word}: {count} ")
