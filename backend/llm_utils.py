import json
import random
import requests
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

class LLMProvider:
    def __init__(self):
        self.provider_type = None
        self.model_name = None
        self._initialize()

    def _initialize(self):
        # Try Gemini first
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.provider_type = "gemini"
                self.model_name = "gemini-1.5-flash"
                print(f"AI Provider initialized: {self.provider_type} (Gemini)")
                return
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")

        # Fallback to Ollama
        try:
            print(f"Connecting to Ollama at {OLLAMA_URL}...")
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                print(f"Ollama models found: {available_models}")
                
                # Try to find the best match for OLLAMA_MODEL
                if OLLAMA_MODEL in available_models:
                    self.model_name = OLLAMA_MODEL
                elif f"{OLLAMA_MODEL}:latest" in available_models:
                    self.model_name = f"{OLLAMA_MODEL}:latest"
                elif available_models:
                    self.model_name = available_models[0]
                
                if self.model_name:
                    self.provider_type = "ollama"
                    print(f"AI Provider initialized: {self.provider_type} ({self.model_name})")
                    return
        except Exception as e:
            print(f"Ollama connection failed: {e}")

        print("Warning: No AI provider (Gemini/Ollama) found. Using fallback templates.")

    def generate_content(self, prompt):
        print(f"--- Sending Prompt to {self.provider_type} ({self.model_name}) ---")
        if self.provider_type == "gemini":
            response = self.gemini_model.generate_content(prompt)
            print(f"--- {self.provider_type} Response Received ---")
            return response.text
        elif self.provider_type == "ollama":
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            }
            try:
                response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
                if response.status_code == 200:
                    text = response.json().get("response", "")
                    print(f"--- {self.provider_type} Response Received ({len(text)} chars) ---")
                    return text
            except Exception as e:
                print(f"Ollama generation failed: {e}")
        return None

# Global AI instance
ai = LLMProvider()

def get_ai_status():
    if not ai.provider_type:
        return "Templates"
    if ai.provider_type == "ollama":
        return f"Ollama ({ai.model_name})"
    return f"Gemini ({ai.model_name})"

# Expanded Sentence structure templates (Fallback)
TEMPLATES = [
    "Captures the same {style} and {trait} that defined {source}, providing a similar {experience} for fans.",
    "Echoes the {style} found in {source}, particularly through its {trait} and {experience}-driven narrative.",
    "A {adj} match for {source}, sharing {trait} while delivering a unique {style} on {experience}.",
    "Mirrors the {experience} of {source} by utilizing {style} and {trait} to drive the story.",
    "Highly recommended for those who loved {source}'s {trait}; it delivers comparable {style} and {experience}.",
    "A spiritual successor to {source} in terms of {style}, prominently featuring {trait} and a familiar {experience}.",
    "Successfully translates the {experience} from {source} into a fresh context, focusing on {trait} and {style}.",
    "社会 {source} fans will appreciate the {style} here, as well as the {trait} that makes for a similar {experience}.",
    "Blends {style} with {trait} to recreate the unique {experience} that made {source} so compelling.",
    "Stands out for its {trait}, which provides the same {style} and {experience} found in {source}.",
    "With a {style} reminiscent of {source}, this film emphasizes {trait} to create a {adj} {experience}.",
    "The {experience} logic here is clearly inspired by {source}, specifically in its use of {style} and {trait}.",
    "While maintaining its own voice, it borrows the {style} from {source} to deliver a {adj} {trait} and {experience}.",
    "An ideal follow-up to {source} because it elevates the {style} and {trait} into a truly {adj} {experience}.",
    "The connection to {source} is palpable through the {trait}, which anchors the {style} and {experience} throughout."
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
IMPORTANT: Do not include any introductory text or markdown formatting. Just the raw JSON.
Example: ["Movie Title 1", "Movie Title 2", ...]
"""
        response_text = ai.generate_content(prompt)
        if not response_text: return None

        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            ranked_titles = json.loads(json_match.group())
            rank_map = {title: idx for idx, title in enumerate(ranked_titles)}
            return sorted(candidates, key=lambda c: rank_map.get(c["title"], 999))
    except Exception as e:
        print(f"AI re-ranking failed: {e}")
    return None

import re

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

For each candidate, write ONE brief sentence (max 15 words) explaining why someone who liked "{source_title}" would enjoy it.
Focus on shared genres, themes, and tone.

Return ONLY a JSON array of strings in the same order as candidates.
IMPORTANT: Ensure each explanation is unique and SPECIFIC to why this movie matches the source. Avoid generic phrases.
Use "INVALID" for candidates with no clear connection.

Example format: ["Reason 1", "Reason 2", ...]
"""
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
                    while len(explanations) < len(matches):
                        explanations.append("INVALID")
                    return explanations[:len(matches)]
        except Exception as e:
            print(f"AI explanation generation failed: {e}, falling back to templates")

    # Fallback to templates
    source_title = source_movie['title']
    source_genres = set(source_movie['genres'].split('|'))
    used_templates, used_vocab, reasons = set(), set(), []

    for target in matches:
        target_genres = set(target['genres'].split('|'))
        shared = source_genres & target_genres
        if not shared and target.get('final_score', 0) < 0.6:
            reasons.append("INVALID"); continue

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

        reasons.append(template.format(style=pick_unique(v['style']), trait=pick_unique(v['trait']), experience=pick_unique(v['experience']), source=source_title, adj=pick_unique(v['adj'])))
    return reasons
