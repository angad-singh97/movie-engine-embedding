from engine import MovieSearchEngine

def run_tests():
    engine = MovieSearchEngine()

    test_queries = [
        "movies directed by christopher nolan",
        "harry potter",
        "sci-fi with tom cruise",
        "animated films with animals"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")

        #you have the three query methods available to use here @Prabal

        print("\n📘 TF-IDF Results:")
        for i, (title, year, genres, score) in enumerate(engine.query_tfidf(query), 1):
            print(f"{i}. {title} ({year}) - {genres} [score: {score:.4f}]")

        print("\n🧠 Vanilla SBERT Results:")
        for i, (title, year, genres, score) in enumerate(engine.query_sbert_vanilla(query), 1):
            print(f"{i}. {title} ({year}) - {genres} [score: {score:.4f}]")

        print("\n🔥 Fine-Tuned SBERT Results:")
        for i, (title, year, genres, score) in enumerate(engine.query_sbert_finetuned(query), 1):
            print(f"{i}. {title} ({year}) - {genres} [score: {score:.4f}]")

if __name__ == "__main__":
    run_tests()

