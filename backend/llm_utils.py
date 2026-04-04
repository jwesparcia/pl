import json
import random
import re
import time
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
AI_PRIORITY = os.getenv("AI_PRIORITY", "gemini").lower()


class LLMProvider:
    def __init__(self):
        self.provider_type = None
        self.model_name = None
        self.gemini_client = None
        self._last_error = None  # Track last error for status reporting
        self._failed_providers = set()  # Track providers that persistently fail
        self._initialize()

    def _initialize(self):
        """Try providers in order based on preference: AI_PRIORITY"""
        
        # Determine check order
        check_order = ["openrouter", "gemini", "ollama"]
        if AI_PRIORITY == "ollama":
            check_order = ["ollama", "gemini", "openrouter"]
        elif AI_PRIORITY == "gemini":
            check_order = ["gemini", "openrouter", "ollama"]

        for provider in check_order:
            if self.provider_type: break # Already found a primary

            if provider == "openrouter":
                if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_key_here":
                    self.provider_type = "openrouter"
                    self.model_name = OPENROUTER_MODEL or "google/gemini-2.0-flash-lite-preview-02-05:free"
                    print(f"AI Provider initialized: {self.provider_type} ({self.model_name})")

            elif provider == "gemini":
                if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_key_here":
                    try:
                        from google import genai
                        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                        self.provider_type = "gemini"
                        self.model_name = "gemini-2.0-flash"
                        print(f"AI Provider initialized: {self.provider_type} ({self.model_name})")
                    except Exception as e:
                        print(f"Gemini initialization failed: {e}")

            elif provider == "ollama":
                try:
                    resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
                    if resp.status_code == 200:
                        models = [m['name'] for m in resp.json().get('models', [])]
                        found = next((m for m in models if OLLAMA_MODEL in m), None)
                        if found or not self.provider_type:
                            self.provider_type = "ollama"
                            self.model_name = found or OLLAMA_MODEL
                            print(f"AI Provider initialized: {self.provider_type} ({self.model_name})")
                except:
                    pass

        # Final check for Ollama as fallback if not primary
        if self.provider_type != "ollama":
             try:
                resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1)
                if resp.status_code == 200:
                    print("Ollama detected (available as fallback).")
             except: pass

        if not self.provider_type:
            print("No AI Provider initialized. Using template engine.")

        if not self.provider_type:
            print("No AI Provider initialized. Using template engine.")

    def generate_content(self, prompt: str, max_retries=1) -> str:
        """Central hub for AI generation with auto-fallback between providers.
        
        Uses fast-fail strategy: if a provider fails with auth/quota errors,
        it's marked as failed and skipped on subsequent calls this session.
        """
        
        # Determine the order of attempts based on preference and priority
        providers_to_try = []
        if AI_PRIORITY == "ollama":
            providers_to_try = ["ollama", "gemini", "openrouter"]
        elif self.provider_type == "openrouter":
            providers_to_try = ["openrouter", "gemini", "ollama"]
        elif self.provider_type == "gemini":
            providers_to_try = ["gemini", "openrouter", "ollama"]
        else:
            providers_to_try = ["gemini", "openrouter", "ollama"]

        # Filter out providers that have persistently failed
        active_providers = [p for p in providers_to_try if p not in self._failed_providers]
        
        if not active_providers:
            print("--- All AI providers have failed. Using templates. ---")
            return None

        print(f"--- Starting generation chain: {active_providers} ---")

        for provider in active_providers:
            try:
                if provider == "gemini" and self.gemini_client:
                    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_key_here":
                        for attempt in range(max_retries + 1):
                            try:
                                print(f"--- Attempting Gemini (gemini-2.0-flash) [attempt {attempt+1}] ---")
                                response = self.gemini_client.models.generate_content(
                                    model="gemini-2.0-flash",
                                    contents=prompt
                                )
                                if response and response.text:
                                    print("--- Gemini Success ---")
                                    self._last_error = None
                                    # Clear from failed list on success
                                    self._failed_providers.discard("gemini")
                                    return response.text
                            except Exception as e:
                                error_str = str(e)
                                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                                    # Rate limited — short wait, then try once more
                                    retry_match = re.search(r'retry in (\d+\.?\d*)', error_str.lower())
                                    wait_time = float(retry_match.group(1)) if retry_match else 3.0
                                    wait_time = min(wait_time, 5)  # Cap at 5s to avoid long hangs
                                    if attempt < max_retries:
                                        print(f"Gemini rate limited. Waiting {wait_time:.1f}s before retry {attempt+2}...")
                                        time.sleep(wait_time)
                                        continue
                                    else:
                                        print(f"Gemini rate limit exhausted. Marking as failed for this session.")
                                        self._last_error = "rate_limited"
                                        self._failed_providers.add("gemini")
                                elif "daily" in error_str.lower() or "limit: 0" in error_str.lower():
                                    # Daily quota exhausted — don't retry at all
                                    print(f"Gemini daily quota exhausted. Skipping for this session.")
                                    self._last_error = "rate_limited"
                                    self._failed_providers.add("gemini")
                                else:
                                    print(f"Gemini error: {error_str[:200]}")
                                    self._last_error = "gemini_error"
                                break  # Don't retry non-rate-limit errors
                
                if provider == "openrouter" and OPENROUTER_API_KEY and OPENROUTER_API_KEY != "your_openrouter_key_here":
                    import requests
                    model = OPENROUTER_MODEL or "google/gemini-2.0-flash-lite-preview-02-05:free"
                    print(f"--- Attempting OpenRouter ({model}) ---")
                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "http://localhost:5005",
                        "X-Title": "MovieMind",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15)
                    if resp.status_code == 200:
                        print("--- OpenRouter Success ---")
                        self._last_error = None
                        self._failed_providers.discard("openrouter")
                        return resp.json()['choices'][0]['message']['content']
                    else:
                        error_detail = resp.text[:200] if resp.text else "No details"
                        print(f"OpenRouter error: {resp.status_code} - {error_detail}")
                        if resp.status_code == 401:
                            self._last_error = "openrouter_auth"
                            self._failed_providers.add("openrouter")  # Don't retry invalid keys
                        else:
                            self._last_error = f"openrouter_{resp.status_code}"
                
                if provider == "ollama":
                    import requests
                    model = self.model_name or OLLAMA_MODEL
                    print(f"--- Attempting Ollama ({model}) ---")
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
                    if resp.status_code == 200:
                        print("--- Ollama Success ---")
                        self._last_error = None
                        self._failed_providers.discard("ollama")
                        return resp.json()['response']
            except Exception as e:
                print(f"Provider {provider} failed: {e}. Trying next...")
                self._last_error = f"{provider}_error"
                # Mark persistent failures (but not timeouts for Ollama)
                is_timeout = "timeout" in str(e).lower()
                if not (provider == "ollama" and is_timeout):
                    if "connection" in str(e).lower() or "refused" in str(e).lower() or "auth" in str(e).lower():
                        self._failed_providers.add(provider)
        
        return None


# Global AI instance
ai = LLMProvider()


def get_ai_status():
    if not ai.provider_type:
        return "Templates (no AI key configured)"
    
    # Report actual runtime status
    if ai._last_error == "rate_limited":
        return f"Rate Limited — using templates (Gemini quota exhausted)"
    if ai._last_error == "openrouter_auth":
        return f"Auth Error — using templates (OpenRouter key invalid)"
    if ai._last_error and "error" in ai._last_error:
        return f"AI Offline — using templates"
    
    if ai.provider_type == "ollama":
        return f"Ollama ({ai.model_name})"
    if ai.provider_type == "openrouter":
        return f"OpenRouter ({ai.model_name})"
    return f"Gemini ({ai.model_name})"


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
    Scouts IMDB for a movie poster (replaces AI TMDB hallucination).
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


# Expanded Sentence structure templates (Fallback only — no AI key)
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


def analyze_and_rerank(source_movie, candidates):
    if not ai.provider_type or not candidates:
        return None

    try:
        source_title = source_movie["title"]
        source_genres = source_movie["genres"]
        candidate_titles = "\n".join([f"- {c['title']} ({c['genres']})" for c in candidates[:15]])

        prompt = f"""
I need help re-ranking movie recommendations based on semantic similarity.

Source movie: {source_title} ({source_genres})

Candidate movies:
{candidate_titles}

Rank these candidates by how well they match the source movie in terms of:
- Shared genres and themes
- Similar tone and style
- Likely appeal to fans of the source

Return ONLY a JSON array of movie titles in ranked order (best first).
Focus on matching the narrative weight, emotional core, and atmospheric qualities of the source movie. For example, a political thriller should be matched with other investigative or high-stakes dramas, not just anything in the 'Drama' genre.

IMPORTANT: Do not include any introductory text or markdown formatting. Just the raw JSON.
Example: ["Movie Title 1", "Movie Title 2", ...]
"""
        response_text = ai.generate_content(prompt)
        if not response_text:
            return None

        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            ranked_titles = json.loads(json_match.group())
            rank_map = {title: idx for idx, title in enumerate(ranked_titles)}
            return sorted(candidates, key=lambda c: rank_map.get(c["title"], 999))
    except Exception as e:
        print(f"AI re-ranking failed: {e}")
    return None


def get_movie_explanation(source_movie, matches):
    if ai.provider_type:
        try:
            source_title = source_movie['title']
            source_genres = source_movie['genres']
            candidates_str = "\n".join([f"- {m['title']} ({m['genres']})" for m in matches[:10]])

            prompt = f"""
Generate personalized recommendation explanations for movies similar to: "{source_title}" ({source_genres}).

Candidates:
{candidates_str}

For each candidate, write ONE brief sentence (max 20 words) explaining why someone who liked "{source_title}" would enjoy it.
Focus on shared narrative themes, character archetypes, or specific stylistic similarities. 
CRITICAL: Do not use generic phrases like "Echoes the somber reflection", "A spiritual successor", or "Captures the same style". Mention specific qualities shared by both films.

Return ONLY a JSON array of strings in the same order as candidates.
Use "INVALID" for candidates with no clear connection to the source movie.

Example format: ["Reason 1", "Reason 2", ...]
"""
            print(f"DEBUG: Sending explanation prompt for {len(matches)} movies...")
            response_text = ai.generate_content(prompt)
            if response_text:
                # Robust extraction of JSON from response (handling markdown code blocks)
                json_str = response_text
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0]

                json_match = re.search(r'\[[\s\S]*\]', json_str)
                if json_match:
                    explanations = json.loads(json_match.group())
                    print(f"DEBUG: Successfully parsed {len(explanations)} AI explanations.")
                    while len(explanations) < len(matches):
                        explanations.append("INVALID")
                    return explanations[:len(matches)]
                else:
                    print(f"DEBUG: No JSON array found in AI response: {response_text[:100]}...")
            else:
                print("DEBUG: AI generate_content returned None.")
        except Exception as e:
            print(f"AI explanation generation failed: {e}, falling back to templates")
    else:
        print("DEBUG: AI NOT INITIALIZED (ai.provider_type is None)")

    # Fallback to templates (only when no AI provider is available)
    source_title = source_movie['title']
    source_genres = set(source_movie['genres'].split('|'))
    used_templates, used_vocab, reasons = set(), set(), []

    for target in matches:
        target_genres = set(target['genres'].split('|'))
        shared = source_genres & target_genres
        if not shared and target.get('final_score', 0) < 0.6:
            reasons.append("INVALID")
            continue

        viable_cats = [cat for cat in VOCABULARY if cat in source_genres or cat in target_genres] or ["default"]

        def get_best_category(cats):
            shared_cats = [c for c in cats if c in shared]
            targets = shared_cats if shared_cats else cats
            scores = [(sum(1 for x in VOCABULARY[c]['style']+VOCABULARY[c]['trait']+VOCABULARY[c]['experience'] if x not in used_vocab), c) for c in targets]
            return max(scores, key=lambda x: x[0])[1]

        best_cat = get_best_category(viable_cats)
        v = VOCABULARY[best_cat]

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


def get_pure_ai_reco(movie_title, n=10):
    """
    Asks the AI to suggest movies directly without relying on local data.
    Returns a list of dicts: [{"title": "...", "genres": "...", "reason": "..."}]
    """
    prompt = f"""
I loved the movie "{movie_title}". 
Suggest exactly {n} other movies I might like.

For each suggestion, provide:
1. The full movie title (include the year in parentheses if possible).
2. The genres (pipe-separated, e.g. "Action|Thriller").
3. A specific, context-aware reason why it fits {movie_title}.

IMPORTANT: RETURN ONLY A VALID JSON ARRAY OF OBJECTS.
Example format:
[
  {{"title": "Movie A (2020)", "genres": "Drama|Sci-Fi", "reason": "Because..."}},
  ...
]
Do not include any intro or outro text.
"""
    print(f"DEBUG: Requesting Pure AI Recommendations for '{movie_title}'...")
    response_text = ai.generate_content(prompt)
    
    if not response_text:
        print("DEBUG: Pure AI generation returned None.")
        return []

    try:
        # Robust extraction of JSON from response
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]

        json_match = re.search(r'\[[\s\S]*\]', json_str)
        if json_match:
            recommendations = json.loads(json_match.group())
            print(f"DEBUG: Successfully received {len(recommendations)} Pure AI recommendations.")
            return recommendations[:n]
        else:
            print(f"DEBUG: No JSON array found in AI response: {response_text[:100]}...")
    except Exception as e:
        print(f"Pure AI recommendation parsing failed: {e}")
    
    return []
