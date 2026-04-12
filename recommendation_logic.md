# How MovieMind Computes Recommendations

MovieMind uses a sophisticated **Hybrid Intelligent Engine** that combines **Semantic Vector Search**, **Weighted Keyword Extraction**, and **Thematic Intelligence** to provide precise and diverse movie suggestions.

---

## 1. Feature Engineering (The Semantic "Combined" String)
The core of our intelligence lies in how we vectorize movies. In [build_tmdb_metadata.py](file:///e:/movie-recommendation/backend/build_tmdb_metadata.py), we create a `combined` document for each movie by weighting features based on their technical importance:

| Feature | Weight | Role |
| :--- | :--- | :--- |
| **High-Value Themes** | **3x** | Dominant conceptual signal (e.g., "dystopia", "time travel"). |
| **Keywords** | **2x** | Captures specific plot devices and thematic nuances. |
| **Directors / Cast** | **1x** | Connects consistent creators and lead performances. |
| **Genres** | **1x** | Provides a broad atmospheric base. |
| **Overview** | **1x** | Natural language plot context for semantic depth. |

> **Tech Tip**: Every movie is converted into a 384-dimensional vector using the **all-MiniLM-L6-v2** Transformer model, allowing the system to understand *meaning* (e.g., matching "aliens" with "extraterrestrial") instead of just keyword counts.

---

## 2. Hybrid Scoring Formula
When a user searches for a movie, the engine retrieves candidates and calculates a `final_score` using a multi-layered heuristic:

### Base Equation:
`Score = (Semantic_Similarity * 0.45) + (Weighted_Rating * 0.25) + Bonuses - Penalties`

### Components:
1. **Semantic Similarity (45%)**: Mathematical distance between movie vectors.
2. **Weighted Rating (25%)**: Uses the **IMDb Weighted Rating formula** to favor quality movies over poorly rated ones, even if they are technically "similar".
3. **Thematic Bonuses**: 
   - **Subgenre Match (+0.25)**: Intense boost if movies share a specific niche subgenre (e.g., "Slasher", "Soul-swapping").
   - **High-Value Theme (+0.12)**: Boost for deep conceptual overlaps.
4. **Mode-based Boosts**: 
   - **Story Mode**: Favors rare thematic keyword overlaps.
   - **Director/Actor Mode**: Favors shared personal connections.
5. **Tone Penalties (-0.20)**: Penalizes major tone clashes (e.g., recommending a goofy Comedy after a gritty Horror) to keep the "vibe" consistent.

---

## 3. Diversity Control (MMR)
To prevent the system from returning 10 identical sequels or clones, we implement **Maximal Marginal Relevance (MMR)**:
- **Relevance**: How similar the movie is to your search.
- **Diversity**: How *different* the recommendation is from other movies already in the top results.
- **Result**: You get a healthy mix of direct sequels, similar thematic stories, and unexpected but conceptually related "wildcards."

---

## 4. Final Re-Ranking
As of the latest update, the system performs a final sort on the MMR selections to ensure the **Highest Rated** recommendations (within the relevant set) always appear at the top.

---

## 5. Intelligence-Driven Explanations
Explanations are generated dynamically by [reco_utils.py](file:///e:/movie-recommendation/backend/reco_utils.py) using a tiered logic:
- **Tier 1 (Conceptual)**: "Dives into deep existential questions about the nature of reality..."
- **Tier 2 (Personnel)**: "Reunites you with director Christopher Nolan to explore..."
- **Tier 3 (Keywords)**: "Connected through shared themes of [betrayal and revenge]..."
