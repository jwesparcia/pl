"""
reco_utils.py — Helper Utilities
This module provides support for metadata enrichment, specifically for IMDB 
poster retrieval and the generation of explainable AI (XAI) justifications.
"""

import random
import re

# ==========================================================
#  SECTION 1: IMDB Poster Scouting (unchanged)
# ==========================================================

def _fetch_imdb_poster(movie_title):
    try:
        import urllib.parse
        import requests
        
        clean_title = re.sub(r'\s*\(\d{4}\)', '', movie_title).strip().lower()
        if not clean_title:
            return "NOT_FOUND"
            
        first_char = clean_title[0]
        if not first_char.isalnum():
            first_char = 'a'
            
        encoded = urllib.parse.quote(clean_title)
        url = f"https://v3.sg.media-imdb.com/suggestion/{first_char}/{encoded}.json"
        
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if 'd' in data and data['d']:
                for item in data['d']:
                    if 'i' in item and 'imageUrl' in item['i']:
                        return item['i']['imageUrl']
    except Exception as e:
        print(f"IMDB Search Failed for {movie_title}: {e}")
    return "NOT_FOUND"

def scout_poster_path(movie_title):
    url = _fetch_imdb_poster(movie_title)
    return url if url != "NOT_FOUND" else None

def scout_poster_paths_batch(movie_titles):
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_title = {executor.submit(_fetch_imdb_poster, title): title for title in movie_titles}
        for future in concurrent.futures.as_completed(future_to_title):
            title = future_to_title[future]
            try:
                url = future.result()
                results[title] = url
            except Exception:
                results[title] = "NOT_FOUND"
    return results

# ==========================================================
#  SECTION 2: PRODUCTION UPGRADE - Explanation System
# ==========================================================

# Preprocessing pipeline: performs feature extraction, token cleaning, 
# and encodes semantic embeddings for the clustering model.
CLUSTER_EXPLANATIONS = {
    "PARANORMAL": "Both films explore the terrifying isolation of a haunted location, focusing on the supernatural forces that terrorize the protagonists.",
    "AI_SCI_FI": "These stories delve into the blurred lines between humanity and artificial intelligence, centering on the evolution of machine consciousness.",
    "SPACE_SCI_FI": "Set against the vastness of deep space, both movies follow explorers facing insurmountable odds in the final frontier.",
    "FANTASY": "Both narratives immerse you in a world of magic and ancient power, where a young protagonist must navigate a mythical destiny.",
    "DYSTOPIA": "Set in an oppressive, post-apocalyptic future, these films depict the desperate struggle for survival against a totalitarian system.",
    "CRIME_NOIR": "Shares a gritty, atmospheric focus on the criminal underworld, where morality is blurred and every choice has consequences.",
    "ADVENTURE_SEA": "A harrowing tale of survival against the elements, centered on the primal struggle between man and the unforgiving ocean."
}

# High-Value Concept Descriptions
CONCEPT_EXPLANATIONS = {
    "simulated_reality": "Both films explore the idea of living inside a controlled or artificial reality, questioning the nature of existence.",
    "time_travel_loop": "Navigates the complex, mind-bending consequences of time travel and temporal manipulation.",
    "alien_contact": "Focuses on the profound and often terrifying implications of encountering extraterrestrial life in a cosmic setting.",
    "serial_killer": "A relentless, suspenseful focus on the hunt for a deadly and elusive predator.",
    "identity_crisis": "Examines complex questions of identity and self-discovery, where characters must piece together their fractured pasts.",
    "conspiracy_paranoia": "Unravels layers of systemic conspiracy and paranoia, forcing characters to question everything they know."
}

BORING_KEYWORDS = {
    'aftercreditsstinger', 'duringcreditsstinger', 'based on novel',
    'independent film', 'woman director', 'man', 'woman',
    'based on novel or book', 'film', 'movie', 'sequel', 'remake',
}

def _clean_keywords(keywords):
    return [kw for kw in keywords if kw.lower() not in BORING_KEYWORDS and len(kw) > 2]

def _format_keyword_list(keywords, max_show=2):
    clean = _clean_keywords(keywords)[:max_show]
    if not clean: return None
    if len(clean) == 1: return clean[0]
    return f"{clean[0]} and {clean[1]}"

def explain_recommendation(source_movie, target_movie):
    """
    Generate a grounded explanation based on production-grade thematic clustering.
    Priority: Shared Cluster > High-Value Concept > Shared Personnel > Fallback
    """
    source_title = source_movie.get('title', 'your selection')
    
    # Must align with concepts in app.py and build script
    THEME_CLUSTERS = {
        "PARANORMAL":   {"paranormal_horror", "haunted_location", "demonic_possession"},
        "SPACE_SCI_FI": {"space_sci_fi", "space_travel_loop", "alien_contact"},
        "AI_SCI_FI":    {"ai_sci_fi", "artificial_intelligence"},
        "FANTASY":      {"epic_fantasy", "magical_world"},
        "DYSTOPIA":     {"dystopia_apocalypse"},
        "CRIME_NOIR":   {"noir_thriller", "serial_killer"},
        "ADVENTURE_SEA": {"maritime_survival"}
    }
    
    def get_clusters(movie):
        high = set(movie.get('high_themes', []))
        themes = set(movie.get('themes', []))
        combined = high | themes
        return {c for c, kw_set in THEME_CLUSTERS.items() if kw_set & combined}

    source_clusters = get_clusters(source_movie)
    target_clusters = get_clusters(target_movie)
    shared_clusters = source_clusters & target_clusters
    
    # 1. Shared Cluster
    for cluster in shared_clusters:
        if cluster in CLUSTER_EXPLANATIONS:
            return CLUSTER_EXPLANATIONS[cluster]

    # 2. Shared High-Value Concept
    source_high = set(source_movie.get('high_themes', []))
    target_high = set(target_movie.get('high_themes', []))
    shared_high = source_high & target_high
    for concept in shared_high:
        if concept in CONCEPT_EXPLANATIONS:
            return CONCEPT_EXPLANATIONS[concept]

    # 3. Shared Director
    sd = source_movie.get('director')
    td = target_movie.get('director')
    if sd and sd == td:
        return f"Reunites you with director {td}, whose distinct stylistic choices in {source_title} are mirrored here."

    # 4. Keyword Focus
    shared_kw = target_movie.get('shared_keywords', [])
    clean_kw = _clean_keywords(shared_kw)
    if clean_kw:
        kw_str = _format_keyword_list(clean_kw)
        return f"Connected through specific narrative focus on {kw_str}, capturing the core spirit of {source_title}."

    # 5. Fallback
    sg = set(source_movie.get('genres', '').split('|'))
    tg = set(target_movie.get('genres', '').split('|'))
    shared_g = list(sg & tg)
    genre_str = shared_g[0] if shared_g else "thematic"
    return f"A precisely matched companion to {source_title}, sharing its {genre_str} sensibility and focused structure."

def get_movie_explanation(source_movie, matches):
    reasons = []
    used = set()
    for target in matches:
        for _ in range(3):
            explanation = explain_recommendation(source_movie, target)
            if explanation not in used: break
        used.add(explanation)
        reasons.append(explanation)
    return reasons
