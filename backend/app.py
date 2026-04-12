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
from sentence_transformers import SentenceTransformer
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
similarity_matrix = None   # Pre-computed full cosine similarity matrix
model = None               # SentenceTransformer
embeddings_matrix = None   # Pre-computed semantic embeddings

# Latent factor cache (kept for compatibility, unused with TMDB)
MOVIE_EMBEDDINGS = None
MOVIE_BIASES = None
GENRE_IDF = {}

# Global constants for IMDb formula
GLOBAL_MEAN_RATING = 6.0
MIN_VOTES = 500

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

    # Load movies metadata from JSON
    if os.path.exists(MOVIES_JSON_PATH):
        with open(MOVIES_JSON_PATH, "r", encoding="utf-8") as f:
            movies = json.load(f)
            movie_id_map = {m["movieId"]: m for m in movies}

        # Build index mapping directly from the loaded movies
        for idx, m in enumerate(movies):
            movie2idx[m["movieId"]] = idx
            title2idx[m["title"].strip().lower()] = idx

        # ─── IMPROVED: Feature Engineering ────────────────────────────────
        # Uses the pre-built "combined" field from build_tmdb_metadata.py.
        # Structure: genres(×2) + keywords + cast(top3) + director + overview
        # This ensures TF-IDF captures thematic, personnel, and plot signals.
        def preprocess_data(movies_list):
            """Extract pre-built combined features for TF-IDF vectorization."""
            descriptions = []
            for m in movies_list:
                # Use pre-built combined feature if available (from build script)
                combined = m.get('combined', '')
                if not combined:
                    # Fallback: build on the fly for older movies.json formats
                    clean_t = re.sub(r'\s*\(\d{4}\)', '', m['title'])
                    genres = m['genres'].replace('|', ' ')
                    overview = m.get('overview', '')
                    keywords = ' '.join(m.get('keywords', []))
                    cast = ' '.join([c.replace(' ', '') for c in m.get('cast', [])])
                    director = m.get('director', '').replace(' ', '')
                    combined = f"{clean_t} {genres} {genres} {keywords} {cast} {director} {overview}"
                descriptions.append(combined)
            return descriptions

        # ─── IMPROVED: TF-IDF Configuration ───────────────────────────────
        # max_features bumped from 10k → 15k for richer vocabulary coverage.
        # ngram_range=(1,2) captures bigrams like "space opera", "serial killer".
        def build_similarity_matrix(descriptions):
            """Build TF-IDF matrix and pre-compute the full similarity matrix."""
            print("Initializing Content-Based Search Index (TF-IDF) ...")
            vec = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=15000
            )
            matrix = vec.fit_transform(descriptions)
            print(f"  TF-IDF matrix shape: {matrix.shape}")

            # Pre-compute the FULL cosine similarity matrix for O(1) lookups.
            # For 4803 movies this is ~92MB — perfectly fine in memory.
            print("Pre-computing full cosine similarity matrix ...")
            sim_matrix = cosine_similarity(matrix, matrix)
            print("Search Index Ready.")
            return vec, matrix, sim_matrix
        # Initialize TF-IDF for content-based candidate generation
        movie_descriptions = preprocess_data(movies)
        tfidf, tfidf_matrix, similarity_matrix = build_similarity_matrix(movie_descriptions)

        # Initialize Sentence Transformer for query encoding
        print("Loading Semantic Model (all-MiniLM-L6-v2) ...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract embeddings matrix for fast cosine similarity
        if movies and 'embedding' in movies[0]:
            print("Indexing semantic embeddings ...")
            embeddings_matrix = np.array([m['embedding'] for m in movies], dtype=np.float32)
            print(f"  Embeddings shape: {embeddings_matrix.shape}")
        
        # Calculate IMDb formula constants
        if movies:
            all_votes = [m.get('vote_count', 0) for m in movies]
            all_ratings = [m.get('vote_average', 0) for m in movies if m.get('vote_count', 0) > 0]
            GLOBAL_MEAN_RATING = float(np.mean(all_ratings)) if all_ratings else 6.0
            # Use 75th percentile as m (minimum votes required to be considered a top contender)
            MIN_VOTES = float(np.percentile(all_votes, 75)) if all_votes else 500
            print(f"IMDb Constants: Mean={GLOBAL_MEAN_RATING:.2f}, MinVotes={MIN_VOTES:.0f}")

    print(f"Metadata loaded -- {len(user2idx)} users, {len(movie2idx)} movies")
except Exception as e:
    import traceback
    print(f"Error loading system: {e}")
    traceback.print_exc()

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

# ─── REFINED: Ranking & Diversity Helpers ─────────────────────────────────────

def compute_weighted_rating(vote_avg, vote_count, m=500, C=6.0):
    """IMDb-style weighted rating formula (WR) = (v*R + m*C) / (v+m)"""
    return (vote_count * vote_avg + m * C) / (vote_count + m)

def calculate_mmr(query_similarity, doc_similarities, candidates, lambda_param=0.5, top_n=10):
    """
    Maximal Marginal Relevance to balance relevance and diversity.
    query_similarity: array of sims between query and candidates
    doc_similarities: matrix of sims between candidates
    """
    selected = []
    unselected = list(range(len(candidates)))
    
    # Start with the best match
    best_idx = np.argmax(query_similarity)
    selected.append(best_idx)
    unselected.remove(best_idx)
    
    while len(selected) < min(top_n, len(candidates)):
        mmr_scores = []
        for idx in unselected:
            relevance = query_similarity[idx]
            # Max similarity to any already selected item
            redundancy = max([doc_similarities[idx][s] for s in selected])
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((score, idx))
            
        # Select idx with highest MMR score
        next_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected.append(next_idx)
        unselected.remove(next_idx)
        
    return selected


# ─── Health check ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "index_ready": similarity_matrix is not None, 
        "embeddings_ready": embeddings_matrix is not None,
        "movie_count": len(movies)
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
    Fallback Trending recommendations (TMDB lacks user data natively).
    """
    if not movies:
        abort(503, description="Library not loaded.")

    trending = sorted(movies, key=lambda x: x.get("vote_average", 0), reverse=True)[:10]
    final_pool = []
    for i, m in enumerate(trending):
        final_pool.append({
            "id": m["movieId"],
            "title": fix_title(m["title"]),
            "genres": m["genres"],
            "score": float(m.get("vote_average", 0)),
            "predicted_rating": float(m.get("vote_average", 0)) / 2.0,
            "rank": i + 1,
            "reason": "Top trending movie on TMDB."
        })

    return jsonify({
        "recommendations": final_pool
    })


# ─── Cold Start: Recommend by movie title ─────────────────────────────────────
DUMMY_USER_ID = 9999  # Virtual user for cold-start recommendations

# IMPROVED: Minimum quality threshold — reject anything below this.
# 0.22 is stricter than the old 0.20, ensuring higher specificity.
SIMILARITY_THRESHOLD = 0.22
TOP_N = 10  # Number of recommendations to return

# Generic genres that often cause "shallow" similarity (Drama, Comedy, etc.)
GENERIC_GENRES = {"Drama", "Comedy", "Thriller", "Action"}

# Strict INCOMPATIBILITY Rules (If source has key, target CANNOT have value)
INCOMPATIBLE_GENRES = {
    "Horror": {"Family", "Animation", "Comedy", "Romance"},
    "Family": {"Horror", "Crime"},
    "Animation": {"Horror", "Crime"},
    "Documentary": {"Action", "Science Fiction", "Fantasy"}
}

# Thematic Bridge words (Force-Match) - Updated for new metadata
HIGH_VALUE_CONCEPTS = {
    "simulated_reality", "artificial_intelligence", "dystopia", "time_travel", 
    "identity", "demonic_possession", "alien_contact", "haunted_house", "animal_loyalty"
}

# Subgenre definition (must match exactly the ones in metadata script)
SUBGENRES = {
    "paranormal_horror", "slasher_horror", "space_sci_fi", "ai_sci_fi", "animal_drama", "noir_thriller"
}

def recommend_movies(movie_title, movies_list, tfidf_sim_mat, embeddings, user_context=None):
    """
    Highly optimized recommendation engine using:
      - Hybrid Similarity (TF-IDF + Semantic Embeddings)
      - IMDb Weighted Rating
      - MMR (Maximal Marginal Relevance) for Diversity
      - Dynamic Modes (Story, Actor, Director)
    """
    q_norm = normalize_title(movie_title)
    q_raw = movie_title.lower().strip()
    user_context = user_context or {}
    mode = user_context.get('mode', 'default')
    liked_ids = set(user_context.get('liked_ids', []))
    seen_ids = set(user_context.get('seen_ids', []))

    # ─── Step 1: Find source movie ────────────────────────────────────
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
        if score > 0: matches.append((score, i, m))

    if not matches: return None, None
    matches.sort(key=lambda x: (-x[0], len(x[2]['title'])))
    _, source_idx, source_data = matches[0]

    source_info = {
        "id":       int(source_data["movieId"]),
        "title":    fix_title(source_data["title"]),
        "genres":   source_data["genres"],
        "keywords": list(source_data.get('keywords', [])),
        "themes":   source_data.get('themes', []),
        "high_themes": source_data.get('high_themes', []),
        "cast":     source_data.get('cast', []),
        "director": source_data.get('director', ''),
    }

    # ─── Step 2: Calculate Hybrid Similarity ──────────────────────────
    # a) TF-IDF Similarity
    tfidf_sims = tfidf_sim_mat[source_idx]
    
    # b) Semantic Similarity (O(1) matrix mult)
    query_emb = embeddings[source_idx]
    semantic_sims = np.dot(embeddings, query_emb)
    
    # Combined base similarity (50/50 hybrid)
    base_sims = (tfidf_sims * 0.5) + (semantic_sims * 0.5)

    # ─── Step 3: Candidate Scoring (Pre-MMR) ──────────────────────────
    candidates = []
    source_genres = set(source_info['genres'].split('|'))
    source_keywords = set(source_info['keywords'])
    source_cast = set(source_info['cast'])
    source_title_norm = normalize_title(source_info['title'])

    # Score everyone
    for idx, m in enumerate(movies_list):
        if idx == source_idx: continue
        
        m_id = m['movieId']
        if m_id in seen_ids: continue
        
        m_title_norm = normalize_title(m['title'])
        if m_title_norm == source_title_norm: continue

        # 1. Base Hybrid Sim
        sim_score = float(base_sims[idx])
        if sim_score < 0.15: continue # Liberal gate for MMR to filter
        
        # 2. Weighted Rating (IMDb)
        wr = compute_weighted_rating(
            m.get('vote_average', 0), 
            m.get('vote_count', 0), 
            m=MIN_VOTES, C=GLOBAL_MEAN_RATING
        ) / 10.0 # Normalize to [0,1]
        
        target_genres = set(m['genres'].split('|'))
        
        # ─── STRICT FILTER: Incompatible Genres ───────────────────────────
        is_incompatible = False
        for sg in source_genres:
            if sg in INCOMPATIBLE_GENRES and bool(INCOMPATIBLE_GENRES[sg] & target_genres):
                is_incompatible = True
                break
        if is_incompatible: continue
        
        # ─── STRICT FILTER: High-Value Theme Requirement ──────────────────
        target_themes = set(m.get('themes', []))
        target_high_themes = set(m.get('high_themes', []))
        source_themes = set(source_info.get('themes', []))
        source_high_themes = set(source_info.get('high_themes', []))
        
        shared_themes = source_themes & target_themes
        shared_high_themes = source_high_themes & target_high_themes
        
        # If the source has a HIGH priority theme, the target MUST share at least one HIGH theme
        # or be extremely similar in base features
        if source_high_themes and not shared_high_themes and sim_score < 0.25:
            continue
            
        # ─── SCORING ──────────────────────────────────────────────────────
        theme_bonus = 0.0
        must_include_bonus = 0.0
        subgenre_bonus = 0.0
        
        for theme in shared_themes:
            if theme in SUBGENRES:
                subgenre_bonus += 0.25
                must_include_bonus = 0.2
            elif theme in HIGH_VALUE_CONCEPTS or theme in source_high_themes:
                theme_bonus += 0.25
                must_include_bonus = 0.2
            else:
                theme_bonus += 0.1
        
        # 4. Mode-based Overlap Boost
        target_kw = set(m.get('keywords', []))
        shared_kw = source_keywords & target_kw
        target_cast = set(m.get('cast', []))
        shared_cast = source_cast & target_cast
        
        mode_bonus = 0.0
        if mode == 'story':
            # Boost keyword rarity
            rarity_sum = sum([m.get('keyword_rarity', {}).get(kw, 1.0) for kw in shared_kw])
            mode_bonus = min(0.15, rarity_sum / 5000)
        elif mode == 'actor':
            mode_bonus = 0.2 if shared_cast else 0.0
        elif mode == 'director':
            mode_bonus = 0.2 if m.get('director') == source_info['director'] else 0.0

        # 5. Personalization Boost
        pers_bonus = 0.0
        if liked_ids:
            pers_bonus = 0.05 if any(tid in liked_ids for tid in [m_id]) else 0.0

        # 6. Noise & Tone Mismatch Filtering
        genre_penalty = 0.0
        shared_genres = source_genres & target_genres
        
        if shared_genres.issubset(GENERIC_GENRES) and not shared_kw and not shared_themes:
            genre_penalty = -0.15
            
        tone_penalty = 0.0
        if ("Comedy" in source_genres and "Horror" in target_genres) or ("Horror" in source_genres and "Comedy" in target_genres):
            tone_penalty = -0.2
        
        # Final combined score
        final_score = (sim_score * 0.45) + (wr * 0.25) + theme_bonus + must_include_bonus + subgenre_bonus + mode_bonus + pers_bonus + genre_penalty + tone_penalty
        
        candidates.append({
            "idx": idx,
            "id": int(m_id),
            "title": fix_title(m['title']),
            "genres": m['genres'],
            "score": final_score,
            "predicted_rating": round(min(5.0, max(0.5, final_score * 5.0 + 2.5)), 2),
            "keywords": list(target_kw),
            "shared_keywords": list(shared_kw),
            "themes": m.get('themes', []),
            "shared_themes": list(shared_themes),
            "cast": m.get('cast', []),
            "director": m.get('director', ''),
            "reason_code": 'semantic' if semantic_sims[idx] > tfidf_sims[idx] else 'tfidf'
        })

    # Sort and take top 50 for MMR
    candidates.sort(key=lambda x: x['score'], reverse=True)
    potential_pool = candidates[:30]
    
    if not potential_pool: return source_info, []

    # ─── Step 4: MMR Re-ranking ───────────────────────────────────────
    # Sub-matrix of similarities between candidates for MMR
    pool_indices = [c['idx'] for c in potential_pool]
    candidate_embs = embeddings[pool_indices]
    
    # Sim matrix between candidates (semantic based for diversity)
    doc_sims = np.dot(candidate_embs, candidate_embs.T)
    
    # Query sims (hybrid)
    query_sims = np.array([c['score'] for c in potential_pool])
    
    selected_indices = calculate_mmr(query_sims, doc_sims, potential_pool, lambda_param=0.6, top_n=TOP_N)
    final_results = [potential_pool[i] for i in selected_indices]

    return source_info, final_results


@app.route("/recommend-by-movie", methods=["POST"])
def recommend_by_movie():
    """
    Optimized Hybrid Recommendation API.
    """
    body = request.get_json(force=True, silent=True) or {}
    movie_title = body.get("movie_title", "").strip()
    user_context = body.get("user_context", {}) # mode, liked_ids, seen_ids

    if not movie_title:
        abort(400, description="Missing 'movie_title'.")

    if not movies or similarity_matrix is None or embeddings_matrix is None:
        print(f"DEBUG: movies={len(movies) if movies else 0}, sim_mat={similarity_matrix is not None}, emb_mat={embeddings_matrix is not None}")
        abort(503, description="Search index or embeddings not ready. Run build script first.")

    source_info, final_results = recommend_movies(
        movie_title, movies, similarity_matrix, embeddings_matrix, user_context
    )

    if not source_info or not final_results:
        abort(404, description="Movie not found or no recommendations available.")

    # Sort results by rating (Highest first) as per user request
    final_results.sort(key=lambda x: x.get('predicted_rating', 0), reverse=True)

    from reco_utils import get_movie_explanation
    explanations = get_movie_explanation(source_info, final_results)

    resp_reco = []
    for i, r in enumerate(final_results):
        r["rank"] = i + 1
        r["reason"] = explanations[i]
        # Clean internal metadata
        for k in ["keywords", "shared_keywords", "cast", "director", "idx", "reason_code"]:
            r.pop(k, None)
        resp_reco.append(r)

    return jsonify({
        "source_movie": source_info,
        "recommendations": resp_reco,
        "mode": user_context.get('mode', 'default')
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
