from sklearn.metrics.pairwise import cosine_similarity #to compare the embeddings during retrieval
import os
import pickle

def save_sbert_embeddings(embeddings, path="sbert_embeddings.pkl"):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"✅ SBERT embeddings saved to {path}")

def load_sbert_embeddings(path="sbert_embeddings.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            print(f"✅ Loaded SBERT embeddings from {path}")
            return pickle.load(f)
    else:
        return None
    
def search_sbert(query, sbert_model, sbert_embeddings, movies_df, top_k=5):
    query_embedding = sbert_model.encode([query])
    scores = cosine_similarity(query_embedding, sbert_embeddings).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        movie = movies_df.iloc[idx]
        results.append((movie['title'], movie['release_year'], movie['genres'], scores[idx]))

    return results
