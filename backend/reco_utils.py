"""
reco_utils.py — Recommendation Utilities
==========================================================
Contains:
  1. IMDB poster scouting (unchanged)
  2. IMPROVED: Keyword-aware explanation system
     - Compares shared keywords between source and target
     - Detects shared cast/director
     - Falls back to genre-based templates when no keywords match
==========================================================
"""

import random

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: IMDB Poster Scouting (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_imdb_poster(movie_title):
    try:
        import urllib.parse
        import requests
        import re
        
        # Clean title: "Toy Story (1995)" -> "toy story"
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
                        # Return the direct Amazon media URL
                        return item['i']['imageUrl']
    except Exception as e:
        print(f"IMDB Search Failed for {movie_title}: {e}")
    return "NOT_FOUND"


def scout_poster_path(movie_title):
    """
    Scouts IMDB for a movie poster.
    """
    url = _fetch_imdb_poster(movie_title)
    return url if url != "NOT_FOUND" else None


def scout_poster_paths_batch(movie_titles):
    """
    Scouts IMDB posters for multiple movies simultaneously using threads.
    Returns { "Movie Title": "https://m.media-amazon.com/..." }
    """
    import concurrent.futures
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_title = {executor.submit(_fetch_imdb_poster, title): title for title in movie_titles}
        for future in concurrent.futures.as_completed(future_to_title):
            title = future_to_title[future]
            try:
                url = future.result()
                results[title] = url
            except Exception as exc:
                print(f"{title} generated an exception: {exc}")
                results[title] = "NOT_FOUND"
                
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: IMPROVED Explanation System
# ═══════════════════════════════════════════════════════════════════════════════
#
# UPGRADE: The explanation system now uses a 3-tier priority:
#   1. Shared keywords → "Both movies explore themes like [kw1, kw2]"
#   2. Shared cast/director → "Features the same director/actors"
#   3. Genre template fallback → detailed genre-specific language
#
# This produces explanations that feel human-written and Netflix-like,
# rather than generic "similar genre" filler.
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Keyword-based explanation helpers ────────────────────────────────────────

# Keywords that are too generic to be useful in explanations
BORING_KEYWORDS = {
    'aftercreditsstinger', 'duringcreditsstinger', 'based on novel',
    'independent film', 'woman director', 'man', 'woman',
    'based on novel or book', 'film', 'movie', 'sequel', 'remake',
}

def _clean_keywords(keywords):
    """Filter out boring/meta keywords and format for display."""
    return [kw for kw in keywords if kw.lower() not in BORING_KEYWORDS and len(kw) > 2]


def _format_keyword_list(keywords, max_show=3):
    """Format a list of keywords into a natural-language string."""
    clean = _clean_keywords(keywords)[:max_show]
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    elif len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    else:
        return f"{', '.join(clean[:-1])}, and {clean[-1]}"


# ─── Keyword-based explanation templates ──────────────────────────────────────
# These are used when shared keywords exist between source and target.

# ─── High-Intelligence Thematic Explanations ──────────────────────────────────
# Mapping concepts to specific, grounded descriptions (Intelligence Upgrade)
CONCEPT_EXPLANATIONS = {
    # Subgenres
    "paranormal_horror": "Both films follow characters confronting terrifying paranormal forces and hauntings.",
    "slasher_horror": "Shares a relentless, suspenseful focus on surviving a deadly, unstoppable killer.",
    "space_sci_fi": "Explores the vast, isolating, and awe-inspiring environment of deep space exploration.",
    "ai_sci_fi": "Centered on the evolution and consequences of artificial intelligence and the blurred line between man and machine.",
    "animal_drama": "A deeply emotional story anchored by the unbreakable bond between humans and animals.",
    "noir_thriller": "A gritty, atmospheric investigation diving into a morally complex underworld.",
    
    # High-Value Themes
    "simulated_reality": "Both films explore the idea of living inside a controlled or artificial reality, where the main character slowly becomes aware of the truth.",
    "artificial_intelligence": "Centered on the evolution and consequences of artificial intelligence and the blurred line between man and machine.",
    "alien_contact": "Focuses on the profound and often terrifying implications of encountering extraterrestrial life.",
    "demonic_possession": "Plunges into the dark, terrifying battle against demonic possession and malevolent supernatural entities.",
    "haunted_house": "A chilling experience rooted in the inescapable dread of a haunted location.",
    "time_travel": "Navigates the complex, mind-bending consequences of time travel and temporal manipulation.",
    "dystopia": "Both are set in oppressive, dystopian futures where individual freedom is sacrificed for systemic control.",
    "animal_loyalty": "A touching narrative built around the extraordinary loyalty and emotional depth of an animal companion.",
    "existentialism": "Dives into deep existential questions about the nature of reality and the purpose of human existence.",
    
    # Medium-Value Themes
    "surveillance": "Shares a focus on surveillance, the loss of privacy, and the feeling of being watched in an interconnected world.",
    "identity": "Examines complex questions of identity and self-discovery, much like the journey of self-actualization in {source}.",
    "conspiracy": "Unravels layers of conspiracy and systemic paranoia, forcing characters to question everything they know.",
    "cyberpunk": "Shares a gritty, high-tech cyberpunk aesthetic combined with themes of rebellion against corporate control."
}

# Tone Alignment Mapping
TONE_VOCABULARY = {
    "satirical": ["biting satire", "satirical edge", "social commentary"],
    "existential": ["existential weight", "philosophical depth", "introspective journey"],
    "dark": ["darkly atmospheric", "somber gravity", "gritty realism"],
    "introspective": ["thought-provoking introspection", "nuanced character study", "inner turmoil"],
    "noir": ["noir-inspired tension", "moody aesthetics", "moral ambiguity"],
    "surreal": ["dreamlike surrealism", "vivid imagination", "unconventional narrative"]
}

# Tone extraction keywords (Internal use)
TONE_MAP = {
    "satire": "satirical",
    "philosophical": "existential",
    "dark comedy": "satirical",
    "existentialism": "existential",
    "neo noir": "noir",
    "surreal": "surreal",
    "paranoia": "dark"
}

def _get_movie_tone(movie):
    """Extract primary tone from keywords/genres."""
    kw = set(movie.get('keywords', []))
    for key, tone in TONE_MAP.items():
        if key in kw: return tone
    return None

# ──────────────────────────────────────────────────────────────────────────────


# ─── Genre-based fallback templates (kept from original, enhanced) ────────────

GENRE_TEMPLATES = [
    "Captures the same {shared_genres} atmosphere as {source}, with {trait} that drives the story.",
    "A {adj} match for fans of {source}, sharing its {shared_genres} sensibility and {experience}.",
    "Echoes the {shared_genres} spirit of {source} through its {style} and {trait}.",
    "If you enjoyed {source}'s {shared_genres} elements, this offers a similar {experience}.",
    "Mirrors {source}'s {style} within the {shared_genres} space, providing a comparable {experience}.",
    "These movies share similar {shared_genres} DNA and {trait}.",
    "A spiritual successor to {source} with the same {shared_genres} tone.",
]

VOCABULARY = {
    "Action": {
        "style": ["high-octane energy", "kinetic choreography", "visceral intensity", "fast-paced momentum"],
        "trait": ["stunt-driven spectacle", "tactical precision", "heroic stakes", "explosive set-pieces"],
        "experience": ["adrenaline-fueled ride", "gripping confrontation", "spectacular showdown", "intense mission"],
        "adj": ["electrifying", "unrelenting", "blockbuster", "explosive"]
    },
    "Comedy": {
        "style": ["absurd physical humor", "fast-paced wit", "satirical edge", "playful irreverence"],
        "trait": ["mentorship dynamics", "comical misunderstandings", "quirky character arcs", "witty dialogue"],
        "experience": ["laugh-out-loud adventure", "whimsical escape", "hilarious journey", "chaotic fun"],
        "adj": ["brilliant", "uproarious", "charming", "clever"]
    },
    "Animation": {
        "style": ["vibrant visual storytelling", "expressive character design", "whimsical wonder", "fluid animation"],
        "trait": ["imaginative world-building", "emotional depth", "fantastical elements", "family-centric heart"],
        "experience": ["enchanting voyage", "magical discovery", "visual feast", "touching narrative"],
        "adj": ["delightful", "captivating", "stunning", "magical"]
    },
    "Drama": {
        "style": ["poignant intimacy", "nuanced character study", "somber reflection", "tonal gravity"],
        "trait": ["thematic resonance", "authentic dialogue", "complex inner turmoil", "moral ambiguity"],
        "experience": ["moving exploration", "profound journey", "thought-provoking drama", "emotional arc"],
        "adj": ["soulful", "masterful", "haunting", "compelling"]
    },
    "Science Fiction": {
        "style": ["visionary scope", "cerebral concept", "technological awe", "speculative depth"],
        "trait": ["existential themes", "metaphysical questions", "futuristic aesthetics", "innovation"],
        "experience": ["mind-bending odyssey", "otherworldly escape", "intellectual voyage", "unique speculation"],
        "adj": ["groundbreaking", "breathtaking", "inventive", "unparalleled"]
    },
    "Sci-Fi": {
        "style": ["visionary scope", "cerebral concept", "technological awe", "speculative depth"],
        "trait": ["existential themes", "metaphysical questions", "futuristic aesthetics", "innovation"],
        "experience": ["mind-bending odyssey", "otherworldly escape", "intellectual voyage", "unique speculation"],
        "adj": ["groundbreaking", "breathtaking", "inventive", "unparalleled"]
    },
    "Horror": {
        "style": ["atmospheric dread", "psychological tension", "unsettling pacing", "visceral shock"],
        "trait": ["supernatural mystery", "survival instincts", "creeping suspense", "macabre imagery"],
        "experience": ["chilling nightmare", "terrifying ordeal", "gripping descent", "haunting encounter"],
        "adj": ["sinister", "macabre", "pulse-pounding", "eerie"]
    },
    "Thriller": {
        "style": ["relentless tension", "calculated suspense", "clever misdirection", "psychological depth"],
        "trait": ["cat-and-mouse dynamics", "plot twists", "moral ambiguity", "paranoia"],
        "experience": ["edge-of-your-seat thriller", "suspenseful journey", "taut mystery", "gripping puzzle"],
        "adj": ["riveting", "tense", "masterful", "razor-sharp"]
    },
    "Romance": {
        "style": ["tender intimacy", "passionate chemistry", "emotional warmth", "bittersweet longing"],
        "trait": ["love story", "character chemistry", "emotional vulnerability", "heartfelt moments"],
        "experience": ["touching love story", "romantic journey", "heartwarming tale", "emotional rollercoaster"],
        "adj": ["sweeping", "tender", "heartfelt", "enchanting"]
    },
    "Adventure": {
        "style": ["epic scope", "sweeping landscapes", "daring escapism", "grand-scale storytelling"],
        "trait": ["quest-driven narrative", "discovery", "world-building", "heroic journey"],
        "experience": ["thrilling expedition", "epic quest", "grand adventure", "journey of discovery"],
        "adj": ["epic", "sweeping", "magnificent", "exhilarating"]
    },
    "Crime": {
        "style": ["gritty realism", "noir atmosphere", "street-level tension", "morally complex storytelling"],
        "trait": ["criminal underworld", "corruption", "heist dynamics", "moral grey areas"],
        "experience": ["crime saga", "taut investigation", "underworld drama", "complex thriller"],
        "adj": ["gritty", "unflinching", "methodical", "intense"]
    },
    "Fantasy": {
        "style": ["mythical grandeur", "otherworldly beauty", "magical realism", "enchanting atmosphere"],
        "trait": ["world-building", "mythological elements", "magical systems", "epic battles"],
        "experience": ["fantastical journey", "mythical voyage", "enchanting quest", "epic tale"],
        "adj": ["enchanting", "mythical", "spellbinding", "wondrous"]
    },
    "War": {
        "style": ["brutal realism", "somber gravity", "battlefield intensity", "unflinching honesty"],
        "trait": ["camaraderie under fire", "sacrifice", "moral complexity", "survival"],
        "experience": ["harrowing journey", "powerful war drama", "unforgettable saga", "emotional battlefield"],
        "adj": ["powerful", "harrowing", "unforgettable", "devastating"]
    },
    "default": {
        "style": ["narrative depth", "cinematic atmosphere", "stylized direction", "tonal consistency"],
        "trait": ["character development", "thematic resonance", "pacing", "cinematography"],
        "experience": ["immersion", "storytelling", "emotional arc", "viewing experience"],
        "adj": ["solid", "noteworthy", "strong", "distinctive"]
    }
}


# ─── Main Explanation Generator ───────────────────────────────────────────────

def explain_recommendation(source_movie, target_movie):
    """
    Generate a grounded explanation based on thematic intelligence.
    优先级: Shared Concept > Shared Keywords > Personnel > Tone Fallback
    """
    source_title = source_movie.get('title', 'your selection')
    shared_kw = target_movie.get('shared_keywords', [])
    clean_shared = _clean_keywords(shared_kw)
    source_themes = set(source_movie.get('themes', []))
    target_themes = set(target_movie.get('themes', []))
    shared_themes = source_themes & target_themes
    
    # ─── Tier 1: Shared High-Value Concept (Highest Intelligence) ─────
    # Prioritize certain "super-themes" for more impactful explanations
    priority_order = [
        "paranormal_horror", "slasher_horror", "space_sci_fi", "ai_sci_fi", "animal_drama", "noir_thriller",
        "demonic_possession", "alien_contact", "simulated_reality", "time_travel", "animal_loyalty", 
        "artificial_intelligence", "haunted_house", "dystopia", "existentialism"
    ]
    sorted_shared = sorted(shared_themes, key=lambda x: priority_order.index(x) if x in priority_order else 99)
    
    for theme in sorted_shared:
        if theme in CONCEPT_EXPLANATIONS:
            desc = CONCEPT_EXPLANATIONS[theme].format(source=source_title)
            return desc

    # ─── Tier 2: Shared Personnel + Keywords ──────────────────────────
    source_director = source_movie.get('director', '')
    target_director = target_movie.get('director', '')
    if clean_shared and source_director and source_director == target_director:
        kw_str = _format_keyword_list(clean_shared, max_show=2)
        return (f"Reunites you with director {target_director} to explore similar "
                f"themes of {kw_str}, much like in {source_title}.")

    # ─── Tier 3: Tone & Style Alignment ──────────────────────────────
    source_tone = _get_movie_tone(source_movie)
    target_tone = _get_movie_tone(target_movie)
    if target_tone and (target_tone == source_tone or not source_tone):
        tone_desc = random.choice(TONE_VOCABULARY[target_tone])
        source_genres = set(source_movie.get('genres', '').split('|'))
        target_genres = set(target_movie.get('genres', '').split('|'))
        shared_genres = list(source_genres & target_genres)
        genres_str = shared_genres[0] if shared_genres else "cinematic"
        
        return f"Aligns with {source_title}'s {tone_desc}, delivering a {genres_str} experience that resonates with its core themes."

    # ─── Tier 4: General Keyword Matching ─────────────────────────────
    if len(clean_shared) >= 1:
        kw_str = _format_keyword_list(clean_shared, max_show=2)
        return f"Connected through themes of {kw_str}, matching the conceptual depth found in {source_title}."

    # ─── Tier 5: Fallback Personnel/Genre ────────────────────────────
    source_cast = set(source_movie.get('cast', []))
    target_cast = set(target_movie.get('cast', []))
    shared_actors = source_cast & target_cast
    if shared_actors:
        actor = list(shared_actors)[0]
        return f"Features {actor}, whose work in {source_title} shares a similar creative energy and focus."

    # Final genre fallback
    source_genres = set(source_movie.get('genres', '').split('|'))
    target_genres = set(target_movie.get('genres', '').split('|'))
    shared_list = list(source_genres & target_genres)
    shared_str = shared_list[0] if shared_list else "thematic"
    
    return f"A natural companion to {source_title}, sharing its {shared_str} sensibility and focused narrative."


def get_movie_explanation(source_movie, matches):
    """
    Generate explanations for a list of recommended movies.

    IMPROVED: Now uses the 3-tier explain_recommendation() function
    which prioritizes shared keywords > shared personnel > genre templates.
    This produces more specific, meaningful explanations.

    Args:
        source_movie: dict with keys: title, genres, keywords, cast, director
        matches: list of dicts, each with: title, genres, keywords, shared_keywords, cast, director

    Returns:
        List of explanation strings, one per match.
    """
    reasons = []
    used_explanations = set()

    for target in matches:
        # Generate explanation with retry to avoid exact duplicates
        for _ in range(3):
            explanation = explain_recommendation(source_movie, target)
            if explanation not in used_explanations:
                break

        used_explanations.add(explanation)
        reasons.append(explanation)

    return reasons
