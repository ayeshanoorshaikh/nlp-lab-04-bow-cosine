from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documents
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

# Search query
query = ["machine learning algorithms for data"]

# 1. Fit CountVectorizer on documents
vectorizer = CountVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)

# 2. Transform query into numerical vector
query_vector = vectorizer.transform(query)

# Convert vectors to arrays
doc_vectors_array = doc_vectors.toarray()
query_vector_array = query_vector.toarray()

print("Document Vectors:")
print(doc_vectors_array)

print("\nQuery Vector:")
print(query_vector_array)

# 3. Compute cosine similarity
similarity_scores = cosine_similarity(query_vector, doc_vectors)

# 4. Rank documents from highest score to lowest score
ranked_documents = sorted(
    enumerate(similarity_scores[0]),
    key=lambda x: x[1],
    reverse=True
)

print("\nRanked Documents:")

for rank, (doc_index, score) in enumerate(ranked_documents, start=1):
    print(f"\nRank {rank}")
    print(f"Document: {documents[doc_index]}")
    print(f"Cosine Similarity Score: {score:.4f}")