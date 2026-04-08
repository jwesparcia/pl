# ==============================
# FILE: train_model.py
# ==============================
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
import os

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load ratings (you need ratings.csv in /data)
ratings_path = os.path.join(DATA_DIR, "ratings.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")

if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
    print(f"Error: CSV files not found in {DATA_DIR}")
    exit(1)

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# Encode users and movies
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
movie2idx = {m: i for i, m in enumerate(movie_ids)}

ratings['u'] = ratings['userId'].map(user2idx)
ratings['m'] = ratings['movieId'].map(movie2idx)

# Normalize ratings
min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()
ratings['rating_norm'] = (ratings['rating'] - min_rating) / (max_rating - min_rating)

# Model
num_users = len(user2idx)
num_movies = len(movie2idx)

u_in = Input(shape=(1,))
m_in = Input(shape=(1,))

u_emb = Embedding(num_users, 50)(u_in)
m_emb = Embedding(num_movies, 50)(m_in)

u_vec = Flatten()(u_emb)
m_vec = Flatten()(m_emb)

dot = Dot(axes=1)([u_vec, m_vec])
out = Dense(1, activation='sigmoid')(dot)

model = Model([u_in, m_in], out)
model.compile(loss='mse', optimizer=Adam(0.001))

print("Training model...")
model.fit([ratings['u'], ratings['m']], ratings['rating_norm'], epochs=5, batch_size=256)

model.save(os.path.join(MODEL_DIR, "recommender.keras"))

# Save metadata
meta = {
    "user2idx": {str(k): v for k, v in user2idx.items()},
    "movie2idx": {str(k): v for k, v in movie2idx.items()},
    "idx2movie": {str(v): k for k, v in movie2idx.items()},
    "min_rating": float(min_rating),
    "max_rating": float(max_rating)
}

with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Training complete and metadata saved.")
