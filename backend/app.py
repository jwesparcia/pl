"""
=============================================================================
app.py — Flask REST API for the Movie Recommendation System
=============================================================================

Endpoints:
  GET  /health            → check that model is loaded
  GET  /movies            → list all movies
  POST /recommend         → given user_id, return top-10 recommended movies
  POST /recommend-by-movie → cold start: given a movie title, recommend
                             similar movies using embedding transfer
=============================================================================
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request, abort, send_from_directory
from flask_cors import CORS

# ─── Setup ───────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "frontend"))
MODEL_DIR    = os.path.abspath(os.path.join(ROOT_DIR, "..", "model"))

app = Flask(__name__)
CORS(app)


# ─── Serve frontend files (explicit routes to avoid static_url_path conflicts)
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_frontend(filename):
    """Serve frontend assets (style.css, app.js, etc.) if they exist."""
    try:
        return send_from_directory(FRONTEND_DIR, filename)
    except Exception:
        abort(404)

MODEL_PATH = os.path.join(MODEL_DIR, "recommender.pt")
META_PATH  = os.path.join(MODEL_DIR, "metadata.pkl")
DEVICE     = torch.device("cpu")


# ─── Collaborative Filtering Model (must match train_model.py) ───────────────
class CollaborativeFilteringModel(nn.Module):
    """
    Same architecture as in train_model.py.
    We re-define it here so app.py has no import dependency on train_model.py.
    """
    def __init__(self, num_users, num_movies, embed_dim=50):
        super().__init__()
        self.user_embedding  = nn.Embedding(num_users,  embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.user_bias       = nn.Embedding(num_users,  1)
        self.movie_bias      = nn.Embedding(num_movies, 1)

    def forward(self, user_ids, movie_ids):
        u_vec  = self.user_embedding(user_ids)
        m_vec  = self.movie_embedding(movie_ids)
        dot    = (u_vec * m_vec).sum(dim=1)
        u_bias = self.user_bias(user_ids).squeeze(1)
        m_bias = self.movie_bias(movie_ids).squeeze(1)
        return torch.sigmoid(dot + u_bias + m_bias)


# ─── Load model and metadata at startup ──────────────────────────────────────
print("Loading model and metadata ...")
model = None
meta  = None

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CollaborativeFilteringModel(
        num_users  = checkpoint["num_users"],
        num_movies = checkpoint["num_movies"],
        embed_dim  = checkpoint["embed_dim"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Model loaded  ({checkpoint['num_users']} users, {checkpoint['num_movies']} movies)")
except Exception as e:
    print(f"Could not load model: {e}")
    print("    Run  python backend/train_model.py  first.")

try:
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    user2idx   = meta["user2idx"]
    movie2idx  = meta["movie2idx"]
    idx2movie  = meta["idx2movie"]
    movies_df  = meta["movies_df"]
    min_rating = meta["min_rating"]
    max_rating = meta["max_rating"]

    # Build title -> internal movie index lookup for cold-start endpoint
    title2idx = {}
    for orig_id, idx in movie2idx.items():
        row = movies_df[movies_df["movieId"] == orig_id]
        if not row.empty:
            title2idx[row.iloc[0]["title"].strip().lower()] = idx
    print(f"Metadata loaded -- {len(user2idx)} users, {len(movie2idx)} movies")
    print(f"Title lookup built -- {len(title2idx)} titles indexed")
except Exception as e:
    title2idx = {}
    print(f"Could not load metadata: {e}")


def denormalize(pred_norm):
    """Scale sigmoid output [0,1] back to the original 0.5–5.0 rating range."""
    return float(pred_norm) * (max_rating - min_rating) + min_rating


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "meta_loaded": meta is not None})


# ─── List all movies ──────────────────────────────────────────────────────────
@app.route("/movies", methods=["GET"])
def get_movies():
    """Return every movie: [ {id, title, genres}, … ]"""
    if meta is None:
        abort(503, description="Metadata not loaded — train the model first.")
    records = movies_df[["movieId", "title", "genres"]].rename(
        columns={"movieId": "id"}
    ).to_dict(orient="records")
    return jsonify(records)


# ─── Recommend top-10 movies for a user ───────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Body:    { "user_id": <int> }
    Returns: [ { rank, id, title, genres, predicted_rating }, … ] (top 10)

    Steps:
      1. Map user_id → internal embedding index
      2. Build tensors for every movie
      3. Single forward pass through the model
      4. Sort by predicted score, return top 10
    """
    if model is None or meta is None:
        abort(503, description="Model not loaded — run train_model.py first.")

    body    = request.get_json(force=True, silent=True) or {}
    user_id = body.get("user_id")
    if user_id is None:
        abort(400, description="Missing 'user_id'.")

    user_id = int(user_id)

    # Map to internal index (graceful fallback for unknown users)
    if user_id in user2idx:
        user_idx = user2idx[user_id]
    else:
        user_idx = user_id % len(user2idx)

    all_movie_indices = np.array(list(idx2movie.keys()), dtype=np.int64)
    n = len(all_movie_indices)

    user_tensor  = torch.tensor([user_idx] * n, dtype=torch.long, device=DEVICE)
    movie_tensor = torch.tensor(all_movie_indices, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        preds = model(user_tensor, movie_tensor).cpu().numpy()

    top_indices = np.argsort(preds)[::-1][:10]

    results = []
    for rank, pos in enumerate(top_indices):
        movie_idx         = int(all_movie_indices[pos])
        original_movie_id = idx2movie[movie_idx]
        row               = movies_df[movies_df["movieId"] == original_movie_id]
        if row.empty:
            continue
        results.append({
            "rank":             rank + 1,
            "id":               int(original_movie_id),
            "title":            row.iloc[0]["title"],
            "genres":           row.iloc[0]["genres"],
            "predicted_rating": round(denormalize(preds[pos]), 2),
        })

    return jsonify(results)


# ─── Cold Start: Recommend by movie title (dummy user) ────────────────────────
DUMMY_USER_ID = 9999  # Virtual user for cold-start recommendations

@app.route("/recommend-by-movie", methods=["POST"])
def recommend_by_movie():
    """
    Cold Start Recommendation via Embedding Transfer
    =================================================
    Body:    { "movie_title": "Toy Story (1995)" }
    Returns: { source_movie: {...}, recommendations: [{rank, id, title, genres, predicted_rating}, ...] }

    How it works (the "dummy user" approach):
      1. Look up the selected movie's internal index
      2. Extract its LEARNED embedding vector from model.movie_embedding
      3. Use this vector as the "dummy user" embedding
      4. Compute dot_product(dummy_user_emb, all_movie_embs) + movie_biases
      5. Apply sigmoid to get predicted ratings
      6. Sort descending, exclude the input movie, return top 10

    This solves the cold start problem: even a brand-new user with NO history
    can get intelligent recommendations by simply telling us ONE movie they like.
    The model's learned embedding space captures latent features (genre affinity,
    era, tone, etc.) so similar movies cluster together via dot-product similarity.
    """
    if model is None or meta is None:
        abort(503, description="Model not loaded -- run train_model.py first.")

    body = request.get_json(force=True, silent=True) or {}
    movie_title = body.get("movie_title", "").strip()
    if not movie_title:
        abort(400, description="Missing 'movie_title'.")

    # --- Step 1: Resolve title to internal movie index ---
    lookup_key = movie_title.lower()
    if lookup_key not in title2idx:
        # Fuzzy fallback: try partial match
        matches = [t for t in title2idx if lookup_key in t]
        if not matches:
            abort(404, description=f"Movie '{movie_title}' not found in the dataset.")
        lookup_key = matches[0]  # Take the first partial match

    source_movie_idx = title2idx[lookup_key]

    # Find the original movie info for the response
    source_movie_id = idx2movie[source_movie_idx]
    source_row = movies_df[movies_df["movieId"] == source_movie_id]
    source_info = {
        "id":    int(source_movie_id),
        "title": source_row.iloc[0]["title"],
        "genres": source_row.iloc[0]["genres"],
    }

    # Enforce Movie-Driven Signal
    print("\n[DEBUG] --- ENFORCING MOVIE-DRIVEN SIGNAL ---", flush=True)
    print(f"[DEBUG] Dummy User Assigned ID: {DUMMY_USER_ID}", flush=True)
    print(f"[DEBUG] Assigned explicit interaction rating: 5.0", flush=True)
    print(f"[DEBUG] Source Movie: {source_info['title']} (ID: {source_info['id']})", flush=True)
    
    source_genres = set(source_info["genres"].split("|"))

    # --- Step 2: Multi-Sample Injection & Similarity Computation ---
    with torch.no_grad():
        all_movie_indices = np.array(list(idx2movie.keys()), dtype=np.int64)
        movie_tensor = torch.tensor(all_movie_indices, dtype=torch.long, device=DEVICE)

        all_movie_embs = model.movie_embedding(movie_tensor)   # (N, embed_dim)
        all_movie_bias = model.movie_bias(movie_tensor).squeeze(1)  # (N,)

        # Extract source movie embedding
        source_movie_emb = model.movie_embedding(
            torch.tensor([source_movie_idx], dtype=torch.long, device=DEVICE)
        )  # shape: (1, embed_dim)

        # Compute raw dot product similarities
        raw_similarities = (source_movie_emb * all_movie_embs).sum(dim=1).cpu().numpy()

        # Min-Max normalize similarities so they range [0, 1] for hybrid scoring
        sim_min = raw_similarities.min()
        sim_max = raw_similarities.max()
        if sim_max > sim_min:
            sim_normalized = (raw_similarities - sim_min) / (sim_max - sim_min)
        else:
            sim_normalized = np.zeros_like(raw_similarities)

        # Find top 5 similar movies (excluding source)
        top_sim_indices = np.argsort(sim_normalized)[::-1]
        top_5_similar_indices = []
        top_5_movies_log = []
        for pos in top_sim_indices:
            idx = int(all_movie_indices[pos])
            if idx == source_movie_idx:
                continue
            
            orig_id = idx2movie[idx]
            r = movies_df[movies_df["movieId"] == orig_id]
            if not r.empty:
                top_5_similar_indices.append(idx)
                top_5_movies_log.append({
                    "title": r.iloc[0]["title"],
                    "similarity": round(float(sim_normalized[pos]), 4)
                })
            if len(top_5_similar_indices) == 5:
                break

        print("\n[DEBUG] --- TOP 5 MOST SIMILAR MOVIES (by embedding) ---", flush=True)
        for i, m in enumerate(top_5_movies_log):
            print(f"  {i+1}. {m['title']} (sim: {m['similarity']})", flush=True)

        # Inject ALL of them into dummy user preferences (mean embedding)
        multi_sample_indices = [source_movie_idx] + top_5_similar_indices
        dummy_user_emb = model.movie_embedding(
            torch.tensor(multi_sample_indices, dtype=torch.long, device=DEVICE)
        ).mean(dim=0, keepdim=True)

        # --- Step 3: Score every movie via dummy user embedding ---
        scores = (dummy_user_emb * all_movie_embs).sum(dim=1) + all_movie_bias
        preds = torch.sigmoid(scores).cpu().numpy()

    # --- Step 4: Hybrid Scoring and Filtering ---
    unsorted_results = []
    for pos in range(len(all_movie_indices)):
        movie_idx = int(all_movie_indices[pos])
        if movie_idx == source_movie_idx:
            continue  # Exclude the selected movie itself

        sim_score = float(sim_normalized[pos])
        
        # Filtering rule: Exclude movies with very low similarity scores
        if sim_score < 0.1:
            continue

        original_movie_id = idx2movie[movie_idx]
        row = movies_df[movies_df["movieId"] == original_movie_id]
        if row.empty:
            continue

        pred_score = float(preds[pos])
        predicted_rating = denormalize(pred_score)
        
        # Hybrid Scoring (70% model prediction, 30% similarity score scaled to 5.0)
        sim_score_scaled = sim_score * 5.0
        final_score = (0.7 * predicted_rating) + (0.3 * sim_score_scaled)

        # Prioritize movies with same genre
        movie_genres_str = row.iloc[0]["genres"]
        movie_genres = set(movie_genres_str.split("|"))
        genre_overlap = len(source_genres & movie_genres)
        if genre_overlap > 0:
            final_score += (0.15 * genre_overlap) # Boost based on overlapping genres

        unsorted_results.append({
            "id": int(original_movie_id),
            "title": row.iloc[0]["title"],
            "genres": movie_genres_str,
            "predicted_rating": round(predicted_rating, 2),
            "similarity": round(sim_score, 4),
            "final_score": round(final_score, 4)
        })

    # Sort descending by final_score
    unsorted_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    results = []
    print("\n[DEBUG] --- FINAL RECOMMENDED MOVIE LIST & SCORES ---", flush=True)
    for rank, r in enumerate(unsorted_results[:10]):
        print(f"  {rank+1}. {r['title']} | Final: {r['final_score']} | Pred: {r['predicted_rating']} | Sim: {r['similarity']} | Genres: {r['genres']}", flush=True)
        r["rank"] = rank + 1
        results.append(r)
    print("---------------------------------------------------\n", flush=True)

    return jsonify({
        "source_movie":    source_info,
        "recommendations": results,
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nFlask API ->  http://localhost:5000")
    print("    GET  /movies")
    print("    POST /recommend")
    print("    POST /recommend-by-movie  (cold start)\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
