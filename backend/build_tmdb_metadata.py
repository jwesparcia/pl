import json
import ast
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Preprocessing pipeline for the TMDB 5000 dataset: performs feature extraction, 
# token cleaning, and encodes semantic embeddings for the hybrid recommendation model.

# 1. Fetch TMDB 5000 dataset from Hugging Face
print("Fetching TMDB 5000 dataset from Hugging Face...")
from datasets import load_dataset
ds = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")

# 2. Initialize Sentence Transformer for pre-computing embeddings
print("Initializing Sentence Transformer (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- High-Concept Theme Map (Prioritized Clustering) ---
# Mapping cluster keywords to broader conceptual themes with importance ratings.
THEME_BANK = {
    # HIGH Priority (Strong identifiers for Cluster-based filtering)
    "simulated_reality":        {"priority": "HIGH", "terms": ["simulation", "virtual reality", "simulated reality", "reality show", "videogame", "game world"]},
    "artificial_intelligence":  {"priority": "HIGH", "terms": ["artificial intelligence", "robot", "android", "cyborg", "machine consciousness"]},
    "alien_contact":            {"priority": "HIGH", "terms": ["extraterrestrial", "alien conquest", "first contact", "space war", "alien invasion"]},
    "demonic_possession":       {"priority": "HIGH", "terms": ["demonic", "possession", "devil", "exorcist", "exorcism"]},
    "haunted_location":         {"priority": "HIGH", "terms": ["haunted house", "ghost", "poltergeist", "apparition", "haunting"]},
    "time_travel_loop":         {"priority": "HIGH", "terms": ["time travel", "time loop", "wormhole", "temporal", "time paradox", "future", "past"]},
    "dystopia_apocalypse":      {"priority": "HIGH", "terms": ["dystopia", "dystopian", "totalitarian", "post-apocalyptic", "survival", "cataclysm"]},
    "maritime_survival":        {"priority": "HIGH", "terms": ["shark", "ocean", "sea", "underwater", "shipwreck", "survival at sea"]},
    "magical_world":            {"priority": "HIGH", "terms": ["magic", "wizard", "wand", "witchcraft", "sorcerers", "magical school", "fantasy world"]},
    "serial_killer":            {"priority": "HIGH", "terms": ["serial killer", "maniac", "slasher", "masked killer", "manhunt"]},
    
    # MEDIUM Priority (Thematic depth)
    "surveillance":             {"priority": "MED",  "terms": ["surveillance", "hidden camera", "spy", "voyeur", "voyeurism", "big brother"]},
    "identity_crisis":          {"priority": "MED",  "terms": ["identity", "amnesia", "identity crisis", "doppelganger", "alter ego", "memory loss"]},
    "conspiracy_paranoia":      {"priority": "MED",  "terms": ["conspiracy", "cover-up", "paranoia", "secret society", "corruption"]},
    "coming_of_age":            {"priority": "MED",  "terms": ["coming of age", "teenager", "growing up", "youth", "school life"]},
}

# Explicit Noise/Generic terms to ignore (STRICT Cleaning)
GENERIC_NOISE = {
    'woman director', 'independent film', 'duringcreditsstinger', 'aftercreditsstinger', 
    'based on novel', 'musical', 'biography', '3d', 'new york', 'los angeles',
    'london england', 'nudity', 'family', 'sex', 'funeral', 'party', 'wedding', 'drug',
    'japanese', 'life', 'story', 'love', 'friendship', 'friends', 'city', 'farmhouse',
    'frog', 'village', 'island', 'small town', 'marriage', 'divorce', 'sequel', 'remake'
}

# Subgenre Classification Rules (Refined Clusters)
SUBGENRE_MAP = {
    "paranormal_horror":    {"genres": ["Horror"], "keywords": ["ghost", "demonic", "poltergeist", "apparition", "haunting", "haunted house"]},
    "slasher_horror":       {"genres": ["Horror"], "keywords": ["slasher", "serial killer", "killer"]},
    "space_sci_fi":         {"genres": ["Science Fiction"], "keywords": ["space", "spaceship", "astronaut", "nebula", "wormhole", "planets"]},
    "ai_sci_fi":            {"genres": ["Science Fiction"], "keywords": ["artificial intelligence", "robot", "android", "cyborg"]},
    "epic_fantasy":         {"genres": ["Fantasy", "Adventure"], "keywords": ["magic", "wizard", "wand", "dragon", "quest"]},
    "noir_thriller":        {"genres": ["Thriller", "Crime"], "keywords": ["noir", "gritty", "underworld", "detective", "investigation"]}
}

import re

def classify_subgenres(genres, keywords, overview):
    """Assign specific subgenre tags based on genre+keyword combinations."""
    subgenres = []
    g_set = set(genres)
    k_set = set(keywords)
    ov_lower = overview.lower()
    
    for sub, rules in SUBGENRE_MAP.items():
        # Match if genre matches and it hits a keyword OR a precise word in overview
        if any(g in g_set for g in rules["genres"]):
            if any(k in k_set for k in rules["keywords"]) or any(re.search(rf"\b{k}\b", ov_lower) for k in rules["keywords"]):
                subgenres.append(sub)
    return subgenres

def extract_themes(keywords, overview, genres):
    """Extract prioritized themes and subgenres."""
    themes = {"HIGH": [], "MED": []}
    ov_lower = overview.lower()
    k_set = set(keywords)
    
    # 1. Concept Bank Hits
    for concept, conf in THEME_BANK.items():
        hit = False
        if any(term in k_set for term in conf["terms"]):
            hit = True
        elif any(re.search(rf"\b{term}\b", ov_lower) for term in conf["terms"]):
            hit = True
        
        if hit:
            themes[conf["priority"]].append(concept)

    # 2. Add Subgenres as high-value themes
    subgenres = classify_subgenres(genres, keywords, overview)
    themes["HIGH"].extend(subgenres)

    return themes

# ==========================================================

movies = []
keyword_counts = {}

print("Processing movies and extracting metadata...")
for row in ds:
    try:
        m_id = row['id']
        title = row['title']
        overview = str(row.get('overview', '') or '')
        
        # --- Noise Filtering ---
        # Discard clear low-quality or incomplete metadata
        if len(overview) < 30 or not title:
            continue
        if "Reality-TV" in row.get('genres', '') or "Documentary" in row.get('genres', ''):
            # We keep documentaries but we might handle them differently in app.py
            pass

        vote_average = float(row.get('vote_average', 0.0))
        vote_count = int(row.get('vote_count', 0))
        popularity = float(row.get('popularity', 0.0))
        release_date = str(row.get('release_date', '') or '')

        # --- 1. Parse genres ---
        try:
            genres_raw = ast.literal_eval(row['genres'])
            genre_names = [g['name'] for g in genres_raw] if isinstance(genres_raw, list) else []
        except Exception:
            genre_names = []

        # --- 2. Parse keywords ---
        try:
            kw_raw = ast.literal_eval(row.get('keywords', '[]'))
            keyword_names = [k['name'].lower() for k in kw_raw] if isinstance(kw_raw, list) else []
            # Count keyword frequencies for filtering generic terms
            for kw in keyword_names:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        except Exception:
            keyword_names = []

        # --- 3. Parse cast -> top 3 actors ---
        try:
            cast_raw = ast.literal_eval(row.get('cast', '[]'))
            if isinstance(cast_raw, list):
                sorted_cast = sorted(cast_raw, key=lambda c: c.get('order', 999))
                cast_names = [c['name'].replace(' ', '') for c in sorted_cast[:3]]
            else:
                cast_names = []
        except Exception:
            cast_names = []

        # --- 4. Parse crew -> director only ---
        try:
            crew_raw = ast.literal_eval(row.get('crew', '[]'))
            director = ''
            if isinstance(crew_raw, list):
                for member in crew_raw:
                    if member.get('job') == 'Director':
                        director = member['name'].replace(' ', '')
                        break
        except Exception:
            director = ''

        # --- 5. Extract Prioritized Themes and Subgenres ---
        theme_data = extract_themes(keyword_names, overview, genre_names)
        
        # --- 6. Build "combined" feature string ---
        # Give subgenres and high-themes massive frequency weight
        subgenre_str = ' '.join(theme_data["HIGH"])
        genre_tokens = ' '.join(genre_names)
        keyword_tokens = ' '.join(keyword_names)
        cast_tokens   = ' '.join(cast_names)
        crew_token    = director

        combined = (
            f"{subgenre_str} {subgenre_str} {subgenre_str} "       # Subgenres/High-themes x3
            f"{keyword_tokens} {keyword_tokens} "                  # keywords x2
            f"{cast_tokens} "                                     # cast x1
            f"{crew_token} "                                       # director x1
            f"{genre_tokens} "                                     # genres x1 (base)
            f"{overview}"                                          # plot overview
        ).strip()

        movies.append({
            "movieId":       int(m_id),
            "title":         title,
            "genres":        "|".join(genre_names),
            "overview":      overview,
            "keywords":      keyword_names,
            "themes":        theme_data, # Saved as dict internally, finalized below
            "cast":          cast_names,
            "director":      director,
            "vote_average":  vote_average,
            "vote_count":    vote_count,
            "popularity":    popularity,
            "release_date":  release_date,
            "combined":      combined
        })

    except Exception as e:
        print(f"Skipping row due to error: {e}")

# 3. Pre-compute Semantic Embeddings in batch for O(1) performance
print(f"Computing semantic embeddings for {len(movies)} movies...")
descriptions = [m['combined'] for m in movies]
embeddings = model.encode(descriptions, batch_size=32, show_progress_bar=True)

# 4. Attach embeddings and keyword counts to movies (serializable)
print("Finalizing metadata and theme extraction...")
for i, m in enumerate(movies):
    m['embedding'] = embeddings[i].tolist()
    # Sort keywords by rarity
    rarity_map = {kw: round(len(movies) / keyword_counts.get(kw, 1), 2) for kw in m['keywords']}
    m['keyword_rarity'] = rarity_map
    
    # Refine themes by adding rare but non-noise keywords
    high_themes = m['themes']['HIGH']
    med_themes = m['themes']['MED']
    
    potential_raw = [kw for kw in m['keywords'] if kw not in GENERIC_NOISE and keyword_counts.get(kw, 0) < 40]
    sorted_rare = sorted(potential_raw, key=lambda x: keyword_counts.get(x, 1))
    
    # Consolidate
    final_themes = list(set(high_themes + med_themes))
    # Add rare keywords to themes only if we have room
    for r_kw in sorted_rare:
        if len(final_themes) >= 6: break
        if r_kw not in final_themes:
            final_themes.append(r_kw)
            
    m['themes'] = final_themes
    m['high_themes'] = high_themes # Keep separate for strict ranking

# --- Save to model/movies.json ---
output_path = os.path.join(os.path.dirname(__file__), "..", "model", "movies.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(movies, f, indent=2, ensure_ascii=False)

print(f"Successfully processed {len(movies)} movies into {output_path}")
print(f"  Sample themes for '{movies[0]['title']}': {movies[0]['themes']}")
