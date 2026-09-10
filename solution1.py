from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Customer reviews
corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

# 1. Create CountVectorizer and transform the corpus
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# 2. Extract vocabulary
vocabulary = vectorizer.get_feature_names_out()

print("Vocabulary:")
print(vocabulary)

# 3. Convert sparse matrix into Pandas DataFrame
df_bow = pd.DataFrame(
    X.toarray(),
    columns=vocabulary
)

print("\nBag of Words Matrix:")
print(df_bow)