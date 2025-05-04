from .utils import (
    generate_full_movie_corpus, 
    load_training_examples
)

from .tf_idf import (
    search_tfidf_full,
    tfidf_vectorize_corpus
)

from .sbert import (
    save_sbert_embeddings, 
    load_sbert_embeddings, 
    search_sbert
)

from .constants import (
    VANILLA_MODEL_NAME,
    VANILLA_EMB_PATH,
    FINETUNED_MODEL_DIR,
    FINETUNED_EMB_PATH,
    TRAIN_CSV
)

# CORE PYTHON UTILITIES
import os
import pickle
import re
import json

# DATA HANDLING
import pandas as pd
import numpy as np

# TEXT EMBEDDINGS
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# EVALUATION METRICS
from sklearn.metrics import ndcg_score

# PROGRESS BARS
from tqdm import tqdm

# VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns

# Define your MovieSearchEngine class
class MovieSearchEngine:
    def __init__(self):
        self.movies_df = pd.read_csv("data/merged_movie_data.csv", encoding='utf-8')
        print("Number of movies:", len(self.movies_df))
        print("Columns:", self.movies_df.columns.to_list())
        print(self.movies_df.head(5))

        # Create the corpus and ID map
        self.corpus, self.movie_id_map = generate_full_movie_corpus(self.movies_df)

        # Vectorize with TF-IDF
        self.full_vectorizer, self.full_tfidf_matrix = tfidf_vectorize_corpus(self.corpus)

        # Load Vanilla SBERT
        self.vanilla_model = SentenceTransformer(VANILLA_MODEL_NAME)
        self.vanilla_embeddings = load_sbert_embeddings(VANILLA_EMB_PATH)
        if self.vanilla_embeddings is None:
            print("🔧 Vanilla SBERT embeddings not found. Generating...")
            self.vanilla_embeddings = self.vanilla_model.encode(self.corpus, show_progress_bar=True)
            save_sbert_embeddings(self.vanilla_embeddings, VANILLA_EMB_PATH)
            print("✅ Vanilla SBERT embeddings generated and saved.")
        else:
            print("✅ Vanilla SBERT embeddings loaded from cache.")

        # Fine-tuned SBERT
        if os.path.exists(FINETUNED_MODEL_DIR):
            print(f"✅ Fine-tuned SBERT model found at {FINETUNED_MODEL_DIR}.")
            self.finetuned_model = SentenceTransformer(FINETUNED_MODEL_DIR)
        else:
            print("🚀 Fine-tuning SBERT on custom query-movie pairs...")
            training_data = load_training_examples(TRAIN_CSV, self.movies_df)
            train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)
            self.finetuned_model = SentenceTransformer(VANILLA_MODEL_NAME)
            train_loss = losses.MultipleNegativesRankingLoss(self.finetuned_model)

            self.finetuned_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=2,
                warmup_steps=100,
                show_progress_bar=True
            )
            self.finetuned_model.save(FINETUNED_MODEL_DIR)
            print(f"✅ Fine-tuned model saved to {FINETUNED_MODEL_DIR}")

        # Fine-tuned embeddings
        self.finetuned_embeddings = load_sbert_embeddings(FINETUNED_EMB_PATH)
        if self.finetuned_embeddings is None:
            print("🔧 Generating embeddings using fine-tuned SBERT...")
            self.finetuned_embeddings = self.finetuned_model.encode(self.corpus, show_progress_bar=True)
            save_sbert_embeddings(self.finetuned_embeddings, FINETUNED_EMB_PATH)
            print("✅ Fine-tuned SBERT embeddings saved.")
        else:
            print("✅ Fine-tuned SBERT embeddings loaded from cache.")

    def query_tfidf(self, query, top_k=10):
        return search_tfidf_full(query, self.full_vectorizer, self.full_tfidf_matrix, self.movie_id_map, self.movies_df, top_k)

    def query_sbert_vanilla(self, query, top_k=10):
        return search_sbert(query, self.vanilla_model, self.vanilla_embeddings, self.movies_df, top_k)

    def query_sbert_finetuned(self, query, top_k=10):
        return search_sbert(query, self.finetuned_model, self.finetuned_embeddings, self.movies_df, top_k)
