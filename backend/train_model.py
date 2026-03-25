"""
=============================================================================
train_model.py — Collaborative Filtering with PyTorch (MovieLens)
=============================================================================

WHAT IS COLLABORATIVE FILTERING?
---------------------------------
Collaborative filtering is a recommendation technique that predicts what
a user will like based on the preferences of *similar* users — without
needing to know anything about the movie's content (genre, plot, etc.).

The core idea:
  • If User A and User B both liked movies X and Y,
    and User A also liked movie Z, we recommend Z to User B.

HOW IT WORKS WITH EMBEDDINGS:
-------------------------------
We represent every user and every movie as a small dense vector (embedding).
These vectors are learned during training so that users and movies with
similar taste/style end up with similar vectors.

  User embedding:  [0.3, -0.1, 0.8, ...]  (size: EMBED_DIM)
  Movie embedding: [0.2, -0.3, 0.9, ...]  (size: EMBED_DIM)

DOT PRODUCT SIMILARITY:
-------------------------
The predicted rating is computed as the dot product of the two embeddings
plus bias terms.  A high dot product means the user and movie "align" well.

  score = dot(user_vec, movie_vec) + user_bias + movie_bias

We pass this through a sigmoid function to squash the output to [0, 1],
then scale it back to the [0, 5] rating range at inference time.
=============================================================================
"""

import os
import io
import zipfile
import pickle
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ─── Configuration ───────────────────────────────────────────────────────────
MOVIELENS_URL    = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR        = os.path.join(os.path.dirname(__file__), "..", "model")
EMBED_DIM        = 50       # Size of each user/movie embedding vector
BATCH_SIZE       = 64
EPOCHS           = 10
LR               = 1e-3
VALIDATION_FRAC  = 0.1
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


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
    """
    Encode user IDs and movie IDs to consecutive integers.
    Neural networks need integer indices to look up embeddings.
    """
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


# ─── Step 3: PyTorch Dataset ───────────────────────────────────────────────────
class RatingsDataset(Dataset):
    """Wraps user, movie, and normalized-rating tensors for DataLoader."""
    def __init__(self, users, movies, ratings):
        self.users   = torch.tensor(users,   dtype=torch.long)
        self.movies  = torch.tensor(movies,  dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):  return len(self.ratings)

    def __getitem__(self, i):
        return self.users[i], self.movies[i], self.ratings[i]


# ─── Step 4: Collaborative Filtering Model ────────────────────────────────────
class CollaborativeFilteringModel(nn.Module):
    """
    Embedding-based collaborative filtering model.

    Architecture:
      • User  Embedding (num_users  × embed_dim) — learned user vector
      • Movie Embedding (num_movies × embed_dim) — learned movie vector
      • User  Bias      (num_users  × 1)         — per-user rating offset
      • Movie Bias      (num_movies × 1)         — per-movie rating offset

    Forward pass:
      score = sigmoid( dot(user_vec, movie_vec) + user_bias + movie_bias )

    The sigmoid squashes the raw score to [0, 1]; we scale it to [0–5] at
    inference time.
    """

    def __init__(self, num_users, num_movies, embed_dim=EMBED_DIM):
        super().__init__()

        # ── Embedding tables ─────────────────────────────────────────────────
        # nn.Embedding is a lookup table: given an integer index it returns
        # a learned dense vector of size `embed_dim`.
        self.user_embedding  = nn.Embedding(num_users,  embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)

        # ── Bias terms ───────────────────────────────────────────────────────
        # A bias captures systematic offsets: some users always rate high,
        # some movies are universally loved or disliked.
        self.user_bias  = nn.Embedding(num_users,  1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        # Initialise weights with small values for stable training
        nn.init.normal_(self.user_embedding.weight,  std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_ids, movie_ids):
        """
        Parameters
        ----------
        user_ids  : LongTensor of shape (batch,)
        movie_ids : LongTensor of shape (batch,)

        Returns
        -------
        Tensor of shape (batch,) with predicted ratings in [0, 1].
        """
        # Look up embedding vectors for each user and movie in the batch
        u_vec  = self.user_embedding(user_ids)   # (batch, embed_dim)
        m_vec  = self.movie_embedding(movie_ids)  # (batch, embed_dim)

        # Dot product: element-wise multiply then sum across embed_dim
        # This is the core similarity measure — high dot product ≈ good match
        dot    = (u_vec * m_vec).sum(dim=1)       # (batch,)

        # Add scalar bias for each user and movie
        u_bias = self.user_bias(user_ids).squeeze(1)   # (batch,)
        m_bias = self.movie_bias(movie_ids).squeeze(1) # (batch,)

        # Sigmoid squashes to [0, 1]
        return torch.sigmoid(dot + u_bias + m_bias)    # (batch,)


# ─── Step 5: Train ───────────────────────────────────────────────────────────
def train(df, num_users, num_movies):
    """Split data, build DataLoaders, train and return the best model."""
    # Shuffle and split
    df_shuffled  = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split        = int(len(df_shuffled) * (1 - VALIDATION_FRAC))
    df_train     = df_shuffled.iloc[:split]
    df_val       = df_shuffled.iloc[split:]

    train_ds = RatingsDataset(df_train["user"].values, df_train["movie"].values, df_train["rating_norm"].values)
    val_ds   = RatingsDataset(df_val["user"].values,   df_val["movie"].values,   df_val["rating_norm"].values)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print("\nBuilding model ...")
    model     = CollaborativeFilteringModel(num_users, num_movies, EMBED_DIM).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    criterion = nn.BCELoss()   # Binary cross-entropy for [0,1] targets

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining for {EPOCHS} epochs ...\n")

    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        # ── Training loop ────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for u, m, r in train_loader:
            u, m, r = u.to(DEVICE), m.to(DEVICE), r.to(DEVICE)
            optimizer.zero_grad()
            preds = model(u, m)
            loss  = criterion(preds, r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(r)
        train_loss /= len(train_ds)

        # ── Validation loop ──────────────────────────────────────────────────
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for u, m, r in val_loader:
                u, m, r = u.to(DEVICE), m.to(DEVICE), r.to(DEVICE)
                preds   = model(u, m)
                val_mae += (preds - r).abs().sum().item()
        val_mae /= len(val_ds)

        print(f"  Epoch {epoch:2d}/{EPOCHS}  |  loss: {train_loss:.4f}  |  val_mae: {val_mae:.4f}")

        # Save best model weights
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nTraining complete.  Best Val MAE: {best_val_mae:.4f}")
    return model


# ─── Step 6: Save model + metadata ───────────────────────────────────────────
def save_artifacts(model, user2idx, movie2idx, idx2movie, movies_df,
                   min_rating, max_rating, num_users, num_movies):
    model_path = os.path.join(MODEL_DIR, "recommender.pt")
    meta_path  = os.path.join(MODEL_DIR, "metadata.pkl")

    # Save PyTorch model weights
    torch.save({
        "state_dict": model.state_dict(),
        "num_users":  num_users,
        "num_movies": num_movies,
        "embed_dim":  EMBED_DIM,
    }, model_path)
    print(f"Model saved -> {model_path}")

    metadata = {
        "user2idx":   user2idx,
        "movie2idx":  movie2idx,
        "idx2movie":  idx2movie,
        "movies_df":  movies_df,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "embed_dim":  EMBED_DIM,
        "num_users":  num_users,
        "num_movies": num_movies,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved -> {meta_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ratings_path, movies_path = download_movielens()

    (df, user2idx, movie2idx, idx2movie,
     movies_df, num_users, num_movies,
     min_rating, max_rating) = preprocess(ratings_path, movies_path)

    model = train(df, num_users, num_movies)

    save_artifacts(model, user2idx, movie2idx, idx2movie,
                   movies_df, min_rating, max_rating, num_users, num_movies)

    print("\nDone!  Run 'python app.py' to start the API server.")
