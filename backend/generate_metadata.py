import os
import json
import pandas as pd

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

os.makedirs(MODEL_DIR, exist_ok=True)

def generate():
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    movies_path = os.path.join(DATA_DIR, "movies.csv")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print(f"Error: CSV files not found in {DATA_DIR}")
        return

    print("Preprocessing data...")
    df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)

    user_ids = df["userId"].unique().tolist()
    movie_ids = df["movieId"].unique().tolist()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    min_rating = df["rating"].min()
    max_rating = df["rating"].max()

    print(f"Stats: {len(user_ids)} users, {len(movie_ids)} movies, {len(df)} ratings")

    # Save movies.json
    movies_json_path = os.path.join(MODEL_DIR, "movies.json")
    movies_df.to_json(movies_json_path, orient="records", indent=2)
    print(f"Saved {movies_json_path}")

    # Save metadata.json
    meta = {
        "user2idx": {str(k): v for k, v in user2idx.items()},
        "movie2idx": {str(k): v for k, v in movie2idx.items()},
        "idx2movie": {str(k): v for k, v in idx2movie.items()},
        "min_rating": float(min_rating),
        "max_rating": float(max_rating),
        "num_users": len(user_ids),
        "num_movies": len(movie_ids),
    }
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")

if __name__ == "__main__":
    generate()
