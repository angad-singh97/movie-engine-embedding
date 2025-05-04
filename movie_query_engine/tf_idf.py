from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity #to compare the embeddings during retrieval



def tfidf_vectorize_corpus(corpus):
    print("🔍 Vectorizing full movie texts with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000)  # optional: tune
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print("✅ TF-IDF matrix shape:", tfidf_matrix.shape)
    return vectorizer, tfidf_matrix


def search_tfidf_full(query, vectorizer, tfidf_matrix, movie_id_map, movies_df, top_k=20):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    seen = set()
    for idx in top_indices:
        movie_id = movie_id_map[idx]
        if movie_id not in seen:
            seen.add(movie_id)
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            results.append((movie['title'], movie['release_year'], movie['genres'], scores[idx]))
    
    return results
