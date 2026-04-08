"""
=============================================================================
app.py — Flask REST API for the Movie Recommendation System (Keras)
=============================================================================
"""

import os
import re
import json
import pickle
import threading
import requests
import traceback
from dotenv import load_dotenv

# Load environment variables FIRST to apply Protobuf workaround
load_dotenv()

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed.")
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request, abort, send_from_directory, redirect, Response
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
from reco_utils import get_movie_explanation, scout_poster_path, scout_poster_paths_batch

# ─── Setup ───────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "frontend"))
MODEL_DIR    = os.path.abspath(os.path.join(ROOT_DIR, "..", "model"))

app = Flask(__name__)
CORS(app)


def fix_title(title):
    """Reformat 'Matrix, The (1999)' → 'The Matrix (1999)'."""
    m = re.match(r'^(.+),\s+(The|A|An)\s+(\(\d{4}\))$', title)
    if m:
        return f"{m.group(2)} {m.group(1)} {m.group(3)}"
    return title

def normalize_title(title):
    """Normalize for comparison: remove articles, special chars, AND year."""
    t = title.lower().strip()
    # 1. Remove year: 'matrix (1999)' -> 'matrix'
    t = re.sub(r'\s*\(\d{4}\)\s*$', '', t)
    # 2. Remove articles at start: 'the matrix' -> 'matrix'
    t = re.sub(r'^(the|a|an)\s+', '', t)
    # 3. Remove articles at end of main title: 'matrix, the' -> 'matrix'
    t = re.sub(r',\s+(the|a|an)', '', t)
    # 4. Generic cleanup
    t = re.sub(r'[^a-z0-9\s]', '', t)
    return t

def extract_topics(title):
    """Extract core thematic tokens for semantic matching."""
    # Clean and split
    clean = normalize_title(title)
    tokens = clean.split()
    # Remove common filler words
    stop_words = {'the', 'a', 'an', 'and', 'but', 'or', 'for', 'with', 'on', 'at', 'by', 'of', 'to', 'in', 'is', 'it', 'from'}
    return set([t for t in tokens if t not in stop_words and len(t) > 2])


# ─── Serve frontend index
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

MODEL_PATH     = os.path.join(MODEL_DIR, "recommender.keras")
META_JSON_PATH  = os.path.join(MODEL_DIR, "metadata.json")
MOVIES_JSON_PATH = os.path.join(MODEL_DIR, "movies.json")


# ─── Load Metadata and initialize Search Index ───────────────────────────────
tfidf = None
tfidf_matrix = None
model = None

# Latent factor cache for algorithmic recommendation (Embedded Similarity)
MOVIE_EMBEDDINGS = None 
MOVIE_BIASES = None
GENRE_IDF = {}

try:
    # Pre-initialize variables
    user2idx = {}
    movie2idx = {}
    idx2movie = {}
    movies = []
    movie_id_map = {}
    title2idx = {}
    min_rating = 0.5
    max_rating = 5.0

    # Load scalar metadata from JSON
    if os.path.exists(META_JSON_PATH):
        with open(META_JSON_PATH, "r") as f:
            meta = json.load(f)
            user2idx = {int(k): v for k, v in meta["user2idx"].items()}
            movie2idx = {int(k): v for k, v in meta["movie2idx"].items()}
            idx2movie = {int(k): v for k, v in meta["idx2movie"].items()}
            min_rating = meta.get("min_rating", 0.5)
            max_rating = meta.get("max_rating", 5.0)

    # Load movies metadata from JSON
    if os.path.exists(MOVIES_JSON_PATH):
        with open(MOVIES_JSON_PATH, "r") as f:
            movies = json.load(f)
            movie_id_map = {m["movieId"]: m for m in movies}

        # Build title -> internal movie index lookup
        for orig_id, idx in movie2idx.items():
            m_data = movie_id_map.get(orig_id)
            if m_data:
                title2idx[m_data["title"].strip().lower()] = idx

        def preprocess_data(movies_list):
            """Clean and combine features for TF-IDF."""
            descriptions = []
            for m in movies_list:
                clean_t = re.sub(r'\s*\(\d{4}\)', '', m['title'])
                genres = m['genres'].replace('|', ' ')
                descriptions.append(f"{clean_t} {genres}")
            return descriptions

        def build_similarity_matrix(descriptions):
            """Initialize TF-IDF globally and return the matrix."""
            print("Initializing Content-Based Search Index (TF-IDF) ...")
            vec = TfidfVectorizer(stop_words='english')
            matrix = vec.fit_transform(descriptions)
            print("Search Index Ready.")
            return vec, matrix

        # Initialize TF-IDF for the "Brain" candidate generation
        if SKLEARN_AVAILABLE:
            movie_descriptions = preprocess_data(movies)
            tfidf, tfidf_matrix = build_similarity_matrix(movie_descriptions)

        # Load Keras model
        if os.path.exists(MODEL_PATH):
            print(f"Loading Keras model from {MODEL_PATH}...")
            model = load_model(MODEL_PATH)
            print("Model Loaded.")
            
            # --- EXTRACT LATENT FACTORS FOR ALGORITHM IMPROVEMENT ---
            try:
                # Extract Embeddings (9724, 50) and Biases (9724, 1)
                for layer in model.layers:
                    if layer.name == "movie_embedding":
                        MOVIE_EMBEDDINGS = layer.get_weights()[0]
                    elif layer.name == "movie_bias":
                        MOVIE_BIASES = layer.get_weights()[0]
                print("Latent features (embeddings/biases) cached.")
                
                # --- CALCULATE GENRE IDF ---
                all_genres = []
                for m in movies:
                    all_genres.extend(m['genres'].split('|'))
                
                N = len(movies)
                genre_counts = {}
                for g in all_genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                
                # IDF = log(N / genre_count)
                # This ensures rare genres (Film-Noir) have higher weighting than common ones (Drama)
                GENRE_IDF = {g: np.log(N / count) for g, count in genre_counts.items()}
                print(f"Genre IDF table built for {len(GENRE_IDF)} genres.")
                
            except Exception as e:
                print(f"Warning: Failed to extract latent features from model: {e}")

    print(f"Metadata loaded -- {len(user2idx)} users, {len(movie2idx)} movies")
except Exception as e:
    print(f"Error loading system: {e}")

# Cache and Lock for AI Poster scouting
# Pre-seed with FRESH verified paths (TMDB paths can change frequently!)
poster_cache = {
    "Toy Story (1995)": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
    "Jumanji (1995)": "https://image.tmdb.org/t/p/w500/vgpXmVaVyUL7GGiDeiK1mKEKzcX.jpg",
    "Grumpier Old Men (1995)": "https://image.tmdb.org/t/p/w500/1FSXpj5e8l4KH6nVFO5SPUeraOt.jpg",
    "Waiting to Exhale (1995)": "https://image.tmdb.org/t/p/w500/8MprEuTY3EwkF9nBBPCUyjRjvs5.jpg",
    "Father of the Bride Part II (1995)": "https://image.tmdb.org/t/p/w500/rj4LBtwQ0uGrpBnCELr716Qo3mw.jpg",
    "Heat (1995)": "https://image.tmdb.org/t/p/w500/umSVjVdbVwtx5ryCA2QXL44Durm.jpg",
    "Sabrina (1995)": "https://image.tmdb.org/t/p/w500/i8PbLJDPU7vCwwscWD625oHbJy.jpg",
    "Tom and Huck (1995)": "https://image.tmdb.org/t/p/w500/bMY31ikEOIPOHqQwhPhII8UVBln.jpg",
    "Sudden Death (1995)": "https://image.tmdb.org/t/p/w500/1pylO6YX5XdOA6QCc5IRxrrffkg.jpg",
    "GoldenEye (1995)": "https://image.tmdb.org/t/p/w500/z0ljRnNxIO7CRBhLEO0DvLgAFPR.jpg",
    "The American President (1995)": "https://image.tmdb.org/t/p/w500/yObOAYFIHXHkFPQ3jhgkN2ezaD.jpg",
    "Dracula: Dead and Loving It (1995)": "https://image.tmdb.org/t/p/w500/4rRfZz8YnHNRr16t3CFcJrPdXHi.jpg",
    "Balto (1995)": "https://image.tmdb.org/t/p/w500/dCVcdb5oxDizqFLz0F7TE60NoC9.jpg",
    "Nixon (1995)": "https://image.tmdb.org/t/p/w500/cz2MTGr2wpDZLirgV2rGHBdA2t3.jpg",
    "Casino (1995)": "https://image.tmdb.org/t/p/w500/gziIkUSnYuj9ChCi8qOu2ZunpSC.jpg",
    "Cutthroat Island (1995)": "https://image.tmdb.org/t/p/w500/hYdeBZ4BFXivdouxLfQGWNE6zRx.jpg",
    "Sense and Sensibility (1995)": "https://image.tmdb.org/t/p/w500/cBK2yL3HqhFvIVd7lLtazWlRZPR.jpg",
    "Four Rooms (1995)": "https://image.tmdb.org/t/p/w500/75aHn1NOYXh4M7L5shoeQ6NGykP.jpg",
    "Ace Ventura: When Nature Calls (1995)": "https://image.tmdb.org/t/p/w500/wcinCf1ov2D6M3P7BBZkzQFOiIb.jpg",
    "Money Train (1995)": "https://image.tmdb.org/t/p/w500/jWBDz6Mf9aQVBiUS76JQsEhvoJl.jpg"
}
scout_lock = threading.Lock()
batch_semaphore = threading.Semaphore(3) # Allow 3 parallel batch calls


def denormalize(pred_norm):
    """Scale sigmoid output [0,1] back to the original 0.5–5.0 rating range."""
    return float(pred_norm) * (max_rating - min_rating) + min_rating


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "index_loaded": tfidf_matrix is not None, 
        "meta_loaded": len(movies) > 0
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
    AI-Powered Recommendations for a user.
    """
    if not movies:
        abort(503, description="Library not loaded.")

    body    = request.get_json(force=True, silent=True) or {}
    user_id = body.get("user_id")
    if user_id is None:
        abort(400, description="Missing 'user_id'.")
    
    # Collaborative Filtering Logic
    if not model or not user2idx:
        abort(503, description="Model or User indices not loaded.")

    try:
        user_id_int = int(user_id)
        if user_id_int not in user2idx:
            # Fallback for new users (cold start) -> return top rated movies
            results = sorted(movies, key=lambda x: x.get('avg_rating', 0), reverse=True)[:20]
        else:
            u_idx = user2idx[user_id_int]
            m_indices = list(movie2idx.values())
            m_ids = list(movie2idx.keys())
            
            # Predict for all movies in parallel (vectorized)
            user_input = np.array([u_idx] * len(m_indices))
            movie_input = np.array(m_indices)
            
            preds = model.predict([user_input, movie_input], batch_size=512, verbose=0).flatten()
            
            # Denormalize and pair with movie objects
            results = []
            for i, p in enumerate(preds):
                m_id = m_ids[i]
                m_data = movie_id_map.get(m_id)
                if m_data:
                    results.append({
                        "id": m_id,
                        "title": fix_title(m_data["title"]),
                        "genres": m_data["genres"],
                        "score": float(p),
                        "predicted_rating": round(denormalize(p), 2)
                    })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:50] # Get top 50 candidates
    except Exception as e:
        print(f"Prediction Error: {e}")
        abort(500, description="Internal error during recommendations.")

    source_context = {"title": "Your Profile", "genres": "Various"}
    
    final_pool = results[:10]
    for i, r in enumerate(final_pool):
        r["rank"] = i + 1
        r["reason"] = "Based on your viewing history and preferences."

    return jsonify({
        "recommendations": final_pool
    })


# ─── Cold Start: Recommend by movie title (dummy user) ────────────────────────
DUMMY_USER_ID = 9999  # Virtual user for cold-start recommendations

def recommend_movies(movie_title, movies_list, tfidf_mat):
    """Return top 10 similar movies based purely on TF-IDF cosine similarity."""
    q_norm = normalize_title(movie_title)
    q_raw = movie_title.lower().strip()
    
    matches = []
    for i, m in enumerate(movies_list):
        m_title = m['title'].lower()
        m_fixed = fix_title(m['title']).lower()
        m_norm = normalize_title(m['title'])
        
        score = 0
        if q_raw == m_title or q_raw == m_fixed or (q_norm and q_norm == m_norm):
            score = 3
        elif m_fixed.startswith(q_raw) or (q_norm and m_norm.startswith(q_norm)):
            score = 2
        elif q_raw in m_title or q_raw in m_fixed:
            score = 1
            
        if score > 0:
            matches.append((score, i, m))
            
    if not matches:
        return None, None
        
    matches.sort(key=lambda x: (-x[0], len(x[2]['title'])))
    _, source_idx, source_data = matches[0]
    
    source_info = {
        "id": int(source_data["movieId"]),
        "title": fix_title(source_data["title"]),
        "genres": source_data["genres"]
    }
    
    # TF-IDF Similarity (Cosine)
    query_vec = tfidf_mat[source_idx]
    tfidf_sims = cosine_similarity(query_vec, tfidf_mat).flatten()
    
    # Get top unique candidates
    top_indices = np.argsort(tfidf_sims)[::-1]
    
    candidates = []
    seen_ids = set()
    
    for idx in top_indices:
        if idx == source_idx:
            continue
        m = movies_list[idx]
        m_id = m['movieId']
        if m_id in seen_ids:
            continue
            
        sim_score = float(tfidf_sims[idx])
        if sim_score <= 0.01: # Filter out absolute zero matches
            continue
            
        seen_ids.add(m_id)
        candidates.append({
            "id": int(m["movieId"]),
            "title": fix_title(m["title"]),
            "genres": m["genres"],
            "score": sim_score,
            "predicted_rating": round(min(5.0, max(0.5, sim_score * 5.0 + 2.5)), 2) # Pseudo rating based on score
        })
        
        if len(candidates) >= 10:
            break
            
    return source_info, candidates

@app.route("/recommend-by-movie", methods=["POST"])
def recommend_by_movie():
    """
    Find similar movies using clean TF-IDF Content-Based filtering.
    """
    body = request.get_json(force=True, silent=True) or {}
    movie_title = body.get("movie_title", "").strip()

    if not movie_title:
        abort(400, description="Missing 'movie_title'.")

    if not movies or tfidf_matrix is None:
        abort(503, description="Search index not ready.")

    source_info, final_results = recommend_movies(movie_title, movies, tfidf_matrix)

    if not source_info or not final_results:
        abort(404, description="Movie not found or no recommendations available.")

    # Generate procedural explanations
    from reco_utils import get_movie_explanation as template_explainer
    explanations = template_explainer(source_info, final_results)
    
    resp_reco = []
    for i, r in enumerate(final_results):
        r["rank"] = i + 1
        r["reason"] = explanations[i] if i < len(explanations) else "Related content."
        resp_reco.append(r)

    return jsonify({
        "source_movie": source_info,
        "recommendations": resp_reco
    })

@app.route('/api/posters/batch', methods=['POST'])
def get_posters_batch():
    """
    Scouts multiple posters in one call.
    """
    data = request.json
    titles = data.get('titles', [])
    if not titles:
        return jsonify({})
    
    results = {}
    missing_titles = []
    
    for t in titles:
        if t in poster_cache:
            results[t] = poster_cache[t]
        else:
            missing_titles.append(t)
            
    if missing_titles:
        from reco_utils import scout_poster_paths_batch
        with batch_semaphore:
            batch_results = scout_poster_paths_batch(missing_titles)
            
            # If the AI completely failed (e.g. JSON parse error), don't cache NOT_FOUND
            if not batch_results:
                return jsonify(results)
            
            # The AI might slightly alter the title (e.g. drop the year). 
            # We must map it back to the exact title the frontend requested.
            for mt in missing_titles:
                path = batch_results.get(mt)
                if not path:
                    # Try dropping the year from the requested title to see if the AI returned that
                    import re
                    mt_clean = re.sub(r'\s*\(\d{4}\)', '', mt).strip()
                    path = batch_results.get(mt_clean)
                
                if path and path != "NOT_FOUND":
                    full_url = path
                    poster_cache[mt] = full_url
                    results[mt] = full_url
                else:
                    poster_cache[mt] = "NOT_FOUND"
                    results[mt] = "NOT_FOUND"
    
    return jsonify(results)


@app.route("/api/poster/<path:movie_title>", methods=["GET"])
def get_poster(movie_title):
    """
    Zero-Key Poster Proxy (Full Binary Proxy):
    Uses AI to scout for a TMDB poster, then proxies the ACTUAL IMAGE BYTES.
    """
    movie_title = movie_title.strip()
    target_url = None

    if movie_title in poster_cache:
        target_url = poster_cache[movie_title]
    else:
        with scout_lock:
            # Check again inside lock to see if another thread found it while we waited
            if movie_title in poster_cache:
                target_url = poster_cache[movie_title]
            else:
                from reco_utils import scout_poster_path
                try:
                    scout_title = re.sub(r'\s*\(\d{4}\)$', '', movie_title)
                    print(f"--- Scouting IMDB poster for: {scout_title} ---")
                    path = scout_poster_path(scout_title)
                    if path:
                        target_url = path
                        poster_cache[movie_title] = target_url
                        print(f"Result: {target_url}")
                except Exception as e:
                    print(f"Error scouting poster for {movie_title}: {e}")

    if not target_url:
        return abort(404, description="Poster path not scouted.")

    try:
        print(f"--- Proxying image: {target_url} ---")
        # Proxy the bytes with a standard User-Agent to avoid blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        resp = requests.get(target_url, headers=headers, stream=True, timeout=10)
        if resp.status_code == 200:
            return Response(resp.content, content_type=resp.headers['content-type'])
        return abort(resp.status_code)
    except HTTPException:
        # Re-raise Flask/Werkzeug aborts so they return the correct status code
        raise
    except Exception as e:
        print(f"Error proxying image {target_url}: {e}")
        traceback.print_exc()
        return abort(502)


# ─── Catch-all for assets (Style, JS, Images) ─────────────────────────────────
@app.route("/<path:filename>")
def serve_asset(filename):
    """Serve frontend assets if they exist, after checking all API routes."""
    return send_from_directory(FRONTEND_DIR, filename)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nFlask API ->  http://localhost:5005")
    print("    GET  /movies")
    print("    POST /recommend")
    print("    POST /recommend-by-movie  (cold start)\n")
    app.run(host="0.0.0.0", port=5005, debug=False)
