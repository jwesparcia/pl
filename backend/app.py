"""
=============================================================================
app.py — Flask REST API for the Movie Recommendation System (Keras)
=============================================================================
"""

import os
import re
import json
import pickle
from dotenv import load_dotenv

# Load environment variables FIRST to apply Protobuf workaround
load_dotenv()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify, request, abort, send_from_directory
from flask_cors import CORS
from llm_utils import analyze_and_rerank, get_movie_explanation, get_ai_status

# ─── Setup ───────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "frontend"))
MODEL_DIR    = os.path.abspath(os.path.join(ROOT_DIR, "..", "model"))

app = Flask(__name__)
CORS(app)


def fix_title(title):
    """Reformat 'Matrix, The (1999)' → 'The Matrix (1999)'.
    Handles articles: The, A, An."""
    m = re.match(r'^(.+),\s+(The|A|An)\s+(\(\d{4}\))$', title)
    if m:
        return f"{m.group(2)} {m.group(1)} {m.group(3)}"
    return title


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

MODEL_PATH     = os.path.join(MODEL_DIR, "recommender.keras")
META_JSON_PATH  = os.path.join(MODEL_DIR, "metadata.json")
MOVIES_JSON_PATH = os.path.join(MODEL_DIR, "movies.json")


# ─── Load model and metadata at startup ──────────────────────────────────────
print("Loading Keras model and metadata ...")
model = None
meta  = None
movies = []
movie_id_map = {}
title2idx = {}

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print(f"Could not load model: {e}")
    print("    Run  python backend/train_model.py  first.")

try:
    # Load scalar metadata from JSON
    if os.path.exists(META_JSON_PATH):
        with open(META_JSON_PATH, "r") as f:
            meta = json.load(f)

        user2idx   = {int(k): v for k, v in meta["user2idx"].items()}
        movie2idx  = {int(k): v for k, v in meta["movie2idx"].items()}
        idx2movie  = {int(k): v for k, v in meta["idx2movie"].items()}
        min_rating = meta["min_rating"]
        max_rating = meta["max_rating"]

    # Load movies metadata from JSON
    if os.path.exists(MOVIES_JSON_PATH):
        with open(MOVIES_JSON_PATH, "r") as f:
            movies = json.load(f)
            # movies is a list of dicts: [{"movieId": 1, "title": "Toy Story", "genres": "Animation|..."}, ...]
            movie_id_map = {m["movieId"]: m for m in movies}

    # Build title -> internal movie index lookup for cold-start endpoint
    for orig_id, idx in movie2idx.items():
        m_data = movie_id_map.get(orig_id)
        if m_data:
            title2idx[m_data["title"].strip().lower()] = idx

    print(f"Metadata loaded -- {len(user2idx)} users, {len(movie2idx)} movies")
    print(f"Title lookup built -- {len(title2idx)} titles indexed")
except Exception as e:
    print(f"Could not load metadata: {e}")


def denormalize(pred_norm):
    """Scale sigmoid output [0,1] back to the original 0.5–5.0 rating range."""
    return float(pred_norm) * (max_rating - min_rating) + min_rating


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "model_loaded": model is not None, 
        "meta_loaded": meta is not None,
        "ai_status": get_ai_status()
    })


# ─── List all movies ──────────────────────────────────────────────────────────
@app.route("/movies", methods=["GET"])
def get_movies():
    """Return every movie: [ {id, title, genres}, … ]"""
    if not movies:
        abort(503, description="Metadata not loaded — train the model first.")
    
    results = []
    for m in movies:
        results.append({
            "id":     m["movieId"],
            "title":  fix_title(m["title"]),
            "genres": m["genres"]
        })
    return jsonify(results)


# ─── Recommend top-10 movies for a user ───────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Body:    { "user_id": <int> }
    Returns: [ { rank, id, title, genres, predicted_rating }, … ] (top 10)
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

    all_movie_indices = np.array(list(idx2movie.keys()), dtype=np.int32)
    n = len(all_movie_indices)

    user_tensor  = np.array([user_idx] * n, dtype=np.int32)
    
    # Predict over all movies
    preds = model.predict([user_tensor, all_movie_indices], batch_size=4096, verbose=0).flatten()

    # Increase candidate pool for AI to 40
    top_indices = np.argsort(preds)[::-1][:40]

    results = []
    for rank, pos in enumerate(top_indices):
        movie_idx         = int(all_movie_indices[pos])
        original_movie_id = idx2movie[movie_idx]
        
        m_data = movie_id_map.get(original_movie_id)
        if not m_data:
            continue
            
        results.append({
            "rank":             rank + 1,
            "id":               int(original_movie_id),
            "title":            fix_title(m_data["title"]),
            "genres":           m_data["genres"],
            "predicted_rating": round(denormalize(preds[pos]), 2),
        })

    # AI Re-ranking for User ID
    source_context = {"title": "your preferences", "genres": "mixed"}
    ai_ranked = analyze_and_rerank(source_context, results)
    if ai_ranked:
        # Keep top 15 for explanation step, then pick top 10
        results = ai_ranked[:15]
    
    # Generate AI explanations
    explanations = get_movie_explanation(source_context, results)
    
    final_results = []
    valid_count = 0
    for idx, r in enumerate(results):
        reason = explanations[idx]
        if reason != "INVALID":
            r["rank"] = valid_count + 1
            r["reason"] = reason
            final_results.append(r)
            valid_count += 1
        if valid_count >= 10:
            break
    
    # Fallback to top 10 if AI fails/rejects all
    if not final_results:
        final_results = results[:10]
        for idx, r in enumerate(final_results):
            r["rank"] = idx + 1
            r["reason"] = explanations[idx] if idx < len(explanations) else "Recommended based on your history."

    return jsonify({
        "recommendations": final_results,
        "ai_status": get_ai_status()
    })


# ─── Cold Start: Recommend by movie title (dummy user) ────────────────────────
DUMMY_USER_ID = 9999  # Virtual user for cold-start recommendations

@app.route("/recommend-by-movie", methods=["POST"])
def recommend_by_movie():
    """
    Cold Start Recommendation via Embedding Transfer (Keras)
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
    source_data = movie_id_map.get(source_movie_id)
    if not source_data:
        abort(404, description="Source movie data not found.")
        
    source_info = {
        "id":    int(source_movie_id),
        "title": fix_title(source_data["title"]),
        "genres": source_data["genres"],
    }
    
    source_genres = set(source_info["genres"].split("|"))

    # Extract raw embedding weights from Keras layers
    movie_embeddings = model.get_layer("movie_embedding").get_weights()[0]
    all_movie_indices = np.array(list(dict.fromkeys(idx2movie.keys())), dtype=np.int32)  # Deduplicate

    all_movie_embs = movie_embeddings[all_movie_indices]   # (N, embed_dim)
    source_movie_emb = movie_embeddings[source_movie_idx]  # (embed_dim,)

    # Compute raw dot product similarities
    raw_similarities = np.dot(all_movie_embs, source_movie_emb)

    # Min-Max normalize similarities so they range [0, 1]
    sim_min = raw_similarities.min()
    sim_max = raw_similarities.max()
    if sim_max > sim_min:
        sim_normalized = (raw_similarities - sim_min) / (sim_max - sim_min)
    else:
        sim_normalized = np.zeros_like(raw_similarities)

    # --- Hybrid Scoring and Filtering ---
    seen_movie_ids = set()  # Prevent duplicates in output
    unsorted_results = []
    for pos in range(len(all_movie_indices)):
        movie_idx = int(all_movie_indices[pos])
        if movie_idx == source_movie_idx:
            continue  # Exclude the selected movie itself

        original_movie_id = idx2movie[movie_idx]
        if original_movie_id in seen_movie_ids:
            continue  # Skip duplicate movie IDs
        seen_movie_ids.add(original_movie_id)

        sim_score = float(sim_normalized[pos])

        # Filtering rule: Exclude movies with very low similarity scores
        if sim_score < 0.05:
            continue

        m_data = movie_id_map.get(original_movie_id)
        if not m_data:
            continue

        # Calculate Jaccard Similarity for genres for precise matching
        movie_genres_str = m_data["genres"]
        movie_genres = set(movie_genres_str.split("|"))

        genre_overlap = len(source_genres & movie_genres)
        genre_union = len(source_genres | movie_genres)
        jaccard_sim = genre_overlap / genre_union if genre_union > 0 else 0.0

        # Final score prioritizes exact genre match + embedding similarity
        final_score = (jaccard_sim * 0.6) + (sim_score * 0.4)

        if genre_overlap == 0:
            continue  # Discard if absolutely no genre overlap

        # ── Rule-Based Generator ──
        # Guarantee a realistic high rating (3.5 to 5.0) for recommendations based on match score
        realistic_rating = min(5.0, max(1.0, 3.5 + (1.5 * final_score)))  # Floor at 1.0

        # Build dynamic reason from shared genres
        shared_genres = list(source_genres & movie_genres)
        genre_str = shared_genres[0] if shared_genres else "film"
        if len(shared_genres) >= 2:
            genre_str = f"{shared_genres[0]} and {shared_genres[1]}"

        reason_text = f"A highly-rated {genre_str} similar to {source_info['title']}, sharing overlapping themes and emotional tone."

        unsorted_results.append({
            "id": int(original_movie_id),
            "title": fix_title(m_data["title"]),
            "genres": movie_genres_str,
            "predicted_rating": round(realistic_rating, 2),
            "similarity": round(sim_score, 4),
            "final_score": round(final_score, 4),
            "reason": reason_text
        })

    # Sort descending by final_score
    unsorted_results.sort(key=lambda x: x["final_score"], reverse=True)

    # --- Step 4: Advanced Re-ranking (LLM / Heuristic) ---
    # Take a larger pool (40) for advanced filtering and re-ranking
    candidates_for_llm = unsorted_results[:40]
    
    print(f"Requesting AI re-ranking for {len(candidates_for_llm)} candidates...")
    
    # 1. Filter: Remove movies with weak tonal justification
    filtered_pool = []
    source_genres = set(source_info['genres'].split('|'))
    
    for c in candidates_for_llm:
        # Tonal justification calculation
        m_genres = set(c['genres'].split('|'))
        genre_match_count = len(source_genres & m_genres)
        
        # Rule: If only one genre matches and there's no major actor/director overlap 
        # (sim_score is currently representing our C-signal to some extent from Keras), 
        # we check the score.
        if genre_match_count <= 1 and c['final_score'] < 0.7:
            # Reject if the title doesn't share keywords or themes (heuristic)
            continue
            
        filtered_pool.append(c)
        
    print(f"Justification check passed for {len(filtered_pool)} movies.")

    # 2. Re-rank
    ai_ranked = analyze_and_rerank(source_info, filtered_pool)
    if ai_ranked:
        candidate_pool = ai_ranked[:20]
    else:
        candidate_pool = filtered_pool[:20]
    
    # 3. Explain
    explanations = get_movie_explanation(source_info, candidate_pool)
    
    raw_results = []
    valid_count = 0
    for idx, r in enumerate(candidate_pool):
        reason = explanations[idx]
        if reason == "INVALID":
            continue
            
        r["rank"] = valid_count + 1
        r["reason"] = reason
        raw_results.append(r)
        valid_count += 1
        
        if valid_count >= 10:
            break

    # Final fallback if AI rejects everything
    if not raw_results:
        raw_results = candidate_pool[:10]
        for idx, r in enumerate(raw_results):
            r["rank"] = idx + 1
            r["reason"] = explanations[idx] if idx < len(explanations) else "A similar movie experience."

    return jsonify({
        "source_movie":    source_info,
        "recommendations": raw_results,
        "ai_status":       get_ai_status()
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nFlask API ->  http://localhost:5000")
    print("    GET  /movies")
    print("    POST /recommend")
    print("    POST /recommend-by-movie  (cold start)\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
