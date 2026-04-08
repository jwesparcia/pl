import random

# --- IMDB Poster Scouting ---

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

# --- Procedural Recommendation Explanations ---

TEMPLATES = [
    "Captures the same {style} and {trait} that defined {source}, providing a similar {experience} for fans.",
    "Echoes the {style} found in {source}, particularly through its {trait} and {experience}-driven narrative.",
    "A {adj} match for {source}, sharing {trait} while delivering a unique {style} on {experience}.",
    "Mirrors the {experience} of {source} by utilizing {style} and {trait} to drive the story.",
    "Highly recommended for those who loved {source}'s {trait}; it delivers comparable {style} and {experience}.",
    "A spiritual successor to {source} in terms of {style}, prominently featuring {trait} and a familiar {experience}.",
    "Successfully translates the {experience} from {source} into a fresh context, focusing on {trait} and {style}.",
    "Blends {style} with {trait} to recreate the unique {experience} that made {source} so compelling.",
    "Stands out for its {trait}, which provides the same {style} and {experience} found in {source}.",
    "With a {style} reminiscent of {source}, this film emphasizes {trait} to create a {adj} {experience}.",
    "The {experience} logic here is clearly inspired by {source}, specifically in its use of {style} and {trait}.",
    "While maintaining its own voice, it borrows the {style} from {source} to deliver a {adj} {trait} and {experience}.",
    "An ideal follow-up to {source} because it elevates the {style} and {trait} into a truly {adj} {experience}.",
    "The connection to {source} is palpable through the {trait}, which anchors the {style} and {experience} throughout.",
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
    "default": {
        "style": ["narrative depth", "cinematic atmosphere", "stylized direction", "tonal consistency"],
        "trait": ["character development", "thematic resonance", "pacing", "cinematography"],
        "experience": ["immersion", "storytelling", "emotional arc", "viewing experience"],
        "adj": ["solid", "noteworthy", "strong", "distinctive"]
    }
}


def get_movie_explanation(source_movie, matches):
    """
    Procedural template system for generating movie explanations.
    """
    print("DEBUG: Using Template Fallback for explanations.")
    source_title = source_movie.get('title', 'Your Profile')
    source_genres_str = source_movie.get('genres', 'Default')
    source_genres = set(source_genres_str.split('|'))
    
    used_templates, used_vocab, reasons = set(), set(), []

    for target in matches:
        target_genres = set(target.get('genres', 'Default').split('|'))
        shared = source_genres & target_genres
        
        # Determine the most appropriate vocabulary category
        viable_cats = [cat for cat in VOCABULARY if cat in source_genres or cat in target_genres] or ["default"]

        def get_best_category(cats):
            shared_cats = [c for c in cats if c in shared]
            targets = shared_cats if shared_cats else cats
            # Score based on how many unique words we haven't used yet
            scores = [(sum(1 for x in VOCABULARY[c]['style']+VOCABULARY[c]['trait']+VOCABULARY[c]['experience'] if x not in used_vocab), c) for c in targets]
            return max(scores, key=lambda x: x[0])[1]

        best_cat = get_best_category(viable_cats)
        v = VOCABULARY[best_cat]

        # Pick a fresh template
        available_templates = [t for t in TEMPLATES if t not in used_templates] or list(TEMPLATES)
        template = random.choice(available_templates)
        used_templates.add(template)

        def pick_unique(pool):
            avail = [x for x in pool if x not in used_vocab] or pool
            choice = random.choice(avail)
            used_vocab.add(choice)
            return choice

        reasons.append(template.format(
            style=pick_unique(v['style']),
            trait=pick_unique(v['trait']),
            experience=pick_unique(v['experience']),
            source=source_title,
            adj=pick_unique(v['adj'])
        ))
    return reasons
