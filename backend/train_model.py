"""
=============================================================================
train_model.py — Collaborative Filtering with Keras/TensorFlow (MovieLens)
=============================================================================
"""

import os
import io
import zipfile
import json
import pickle
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ─── Configuration ───────────────────────────────────────────────────────────
MOVIELENS_URL    = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "..", "model")
EMBED_DIM        = 50       # Size of each user/movie embedding vector
BATCH_SIZE       = 64
EPOCHS           = 10
LR               = 1e-3
VALIDATION_FRAC  = 0.1

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Step 1: Download dataset ─────────────────────────────────────────────────
def download_movielens():
    """Download and extract the MovieLens ml-latest-small dataset."""
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    movies_path  = os.path.join(DATA_DIR, "movies.csv")

    if os.path.exists(ratings_path) and os.path.exists(movies_path):
        print("Dataset already downloaded.")
        return ratings_path, movies_path

    print("Downloading MovieLens dataset ...")
    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for name in zf.namelist():
            if name.endswith("ratings.csv") or name.endswith("movies.csv"):
                filename = os.path.basename(name)
                dest = os.path.join(DATA_DIR, filename)
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                print(f"  Extracted -> {dest}")

    print("Dataset ready.")
    return ratings_path, movies_path


# ─── Step 2: Preprocess data ──────────────────────────────────────────────────
def preprocess(ratings_path, movies_path):
    """Encode user IDs and movie IDs to consecutive integers."""
    print("Preprocessing data ...")
    df        = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    user_ids  = df["userId"].unique().tolist()
    movie_ids = df["movieId"].unique().tolist()

    user2idx  = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    df["user"]  = df["userId"].map(user2idx)
    df["movie"] = df["movieId"].map(movie2idx)

    # Normalize ratings to [0, 1] for sigmoid output
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    df["rating_norm"] = (df["rating"] - min_rating) / (max_rating - min_rating)

    num_users  = len(user2idx)
    num_movies = len(movie2idx)
    print(f"  Users: {num_users}  |  Movies: {num_movies}  |  Ratings: {len(df)}")

    return df, user2idx, movie2idx, idx2movie, movies_df, num_users, num_movies, min_rating, max_rating


# ─── Step 3: Keras Model ────────────────────────────────────────────────────
def build_model(num_users, num_movies, embed_dim=EMBED_DIM):
    # Inputs
    user_input = layers.Input(shape=(1,), name="user")
    movie_input = layers.Input(shape=(1,), name="movie")

    # Embeddings
    user_embedding = layers.Embedding(num_users, embed_dim, embeddings_initializer="he_normal", name="user_embedding")(user_input)
    movie_embedding = layers.Embedding(num_movies, embed_dim, embeddings_initializer="he_normal", name="movie_embedding")(movie_input)

    # Flatten embeddings
    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)

    # Dot Product
    dot = layers.Dot(axes=1, name="dot_product")([user_vec, movie_vec])

    # Biases
    user_bias = layers.Embedding(num_users, 1, name="user_bias")(user_input)
    movie_bias = layers.Embedding(num_movies, 1, name="movie_bias")(movie_input)
    
    user_bias = layers.Flatten()(user_bias)
    movie_bias = layers.Flatten()(movie_bias)

    # Add operations
    added = layers.Add(name="add_biases")([dot, user_bias, movie_bias])

    # Sigmoid activation to bound [0, 1]
    output = layers.Activation('sigmoid', name="sigmoid_output")(added)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss="binary_crossentropy", metrics=["mae"])
    return model


# ─── Step 4: Train ───────────────────────────────────────────────────────────
def train(df, num_users, num_movies):
    """Split data, build, train and return Keras model."""
    df_shuffled  = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    users = df_shuffled["user"].values
    movies = df_shuffled["movie"].values
    ratings = df_shuffled["rating_norm"].values

    print("\nBuilding Keras model ...")
    model = build_model(num_users, num_movies, EMBED_DIM)
    model.summary()
    
    # Early Stopping Callback
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    print(f"\nTraining for up to {EPOCHS} epochs ...\n")
    model.fit(
        x=[users, movies],
        y=ratings,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_FRAC,
        callbacks=[early_stop],
        verbose=1
    )

    print(f"\nTraining complete.")
    return model


# ─── Step 5: Save model + metadata ───────────────────────────────────────────
def save_artifacts(model, user2idx, movie2idx, idx2movie, movies_df,
                   min_rating, max_rating, num_users, num_movies):
    model_path = os.path.join(MODEL_DIR, "recommender.keras")
    meta_path  = os.path.join(MODEL_DIR, "metadata.json")

    # Save Keras native model
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    # Save movies_df as CSV (no pickle needed)
    movies_csv_path = os.path.join(MODEL_DIR, "movies_df.csv")
    movies_df.to_csv(movies_csv_path, index=False)
    print(f"Movies CSV saved -> {movies_csv_path}")

    # Save movies_df as JSON (no pandas needed to read)
    movies_json_path = os.path.join(MODEL_DIR, "movies.json")
    movies_df.to_json(movies_json_path, orient="records", indent=2)
    print(f"Movies JSON saved -> {movies_json_path}")

    # Save scalar metadata as JSON
    meta = {
        "user2idx":   {str(k): v for k, v in user2idx.items()},
        "movie2idx":  {str(k): v for k, v in movie2idx.items()},
        "idx2movie":  {str(k): v for k, v in idx2movie.items()},
        "min_rating": float(min_rating),
        "max_rating": float(max_rating),
        "embed_dim":  EMBED_DIM,
        "num_users":  num_users,
        "num_movies": num_movies,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata JSON saved -> {meta_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ratings_path, movies_path = download_movielens()

    (df, user2idx, movie2idx, idx2movie,
     movies_df, num_users, num_movies,
     min_rating, max_rating) = preprocess(ratings_path, movies_path)

    model = train(df, num_users, num_movies)

    save_artifacts(model, user2idx, movie2idx, idx2movie,
                   movies_df, min_rating, max_rating, num_users, num_movies)

    print("\nDone! Run 'python app.py' to start the API server.")
