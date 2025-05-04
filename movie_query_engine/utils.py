def movie_to_vector_string(movie):
    template = (
        "Movie Title: {title} ({release_year}). "
        "Genres: {genres}. "
        "Directed by {director}. "
        "Starring: {top_cast}. "
        "Tagline: '{tagline}'. "
        "Plot: {overview} "
        "Runtime: {runtime} minutes. "
        "Budget: ${budget:,}. Revenue: ${revenue:,}. "
        "Average rating: {vote_average}/10 from {vote_count} votes. "
        "Keywords: {keywords}. "
        "Production: {production_companies}. "
        "Language: {language}."
    )

    def format_field(key, default='Unknown'):
        return str(movie.get(key, default)) or default

    def safe_number(value, fallback=0):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return fallback

    return template.format(
        title=format_field('title'),
        release_year=safe_number(movie.get('release_year'), 'Unknown Year'),
        genres=', '.join(format_field('genres').split('|')),
        director=format_field('director'),
        top_cast=', '.join(format_field('top_cast').split('|')),
        tagline=format_field('tagline', 'No tagline available'),
        overview=format_field('overview', 'No overview available'),
        runtime=safe_number(movie.get('runtime')),
        budget=safe_number(movie.get('budget')),
        revenue=safe_number(movie.get('revenue')),
        vote_average=format_field('vote_average'),
        vote_count=format_field('vote_count'),
        keywords=', '.join(format_field('keywords').split('|')),
        production_companies=', '.join(format_field('production_companies').split('|')),
        language=format_field('language')
    )

def generate_full_movie_corpus(df):
    corpus = []
    movie_id_map = {}
    
    for idx, row in df.iterrows():
        movie_str = movie_to_vector_string(row.to_dict())
        corpus.append(movie_str)
        movie_id_map[len(corpus) - 1] = row['movieId']  # map index to movieId

    return corpus, movie_id_map

def load_training_examples(csv_path, movies_df):
    df = pd.read_csv(csv_path)
    movie_dict = movies_df.set_index("movieId").to_dict(orient="index")
    examples = []

    for _, row in df.iterrows():
        query = row["query"]
        movie_ids = [int(mid) for mid in str(row["movie_ids"]).split(",") if str(mid).strip().isdigit()]
        for mid in movie_ids:
            if mid in movie_dict:
                movie_text = movie_to_vector_string(movie_dict[mid])
                examples.append(InputExample(texts=[query, movie_text], label=1.0))

    print(f"✅ Loaded {len(examples)} training pairs.")
    return examples
