# MovieMind Recommendation Framework

MovieMind utilizes a Hybrid Intelligent Engine that integrates Semantic Vector Search, Statistical Feature Analysis, and Neural Collaborative Filtering to provide accurate and personalized movie suggestions.

---

## 1. Feature Engineering and Vectorization

The system generates a high-dimensional representation of each movie by weighting metadata features according to their thematic significance:

| Feature | Weighting | Role |
| :--- | :--- | :--- |
| **High-Value Themes** | **3.0x** | Significant conceptual identifiers (e.g., dystopia, time travel). |
| **Keywords** | **2.0x** | Specific plot devices and thematic nuances. |
| **Personnel (Cast/Director)** | **1.0x** | Creative and performance connections. |
| **Genres** | **1.0x** | Broad atmospheric categorization. |
| **Narrative Overview** | **1.0x** | Natural language context for semantic depth. |

Technical Note: Movies are encoded into 384-dimensional vectors using the **all-MiniLM-L6-v2** Transformer model. This enables the engine to identify conceptual relationships rather than simple keyword overlaps.

---

## 2. Hybrid Scoring Pipeline

The recommendation process follows a multi-stage scoring architecture to determine the final ranking of candidates:

### Baseline Score Calculation
The initial score is derived from a combination of content-based similarity and global popularity:

`Score = (Semantic_Similarity * 0.45) + (Weighted_IMDb_Rating * 0.15) + Thematic_Bonuses - Tone_Penalties`

### Neural Re-Ranking Layer (Keras 3)
After the baseline pool is generated, the system performs a final scoring pass using a Keras Neural Network:
- **Neural Prediction**: The system computes a predicted rating for the candidate movie based on a collaborative filtering model trained on normalized user rating data.
- **Score Refinement**: The final ranking is adjusted by integrating the neural signal (weighted at 10%) with the hybrid baseline score.

---

## 3. Diversity and Marginal Relevance (MMR)

To ensure that results are not repetitive (e.g., preventing a list of only sequels), the engine implements **Maximal Marginal Relevance (MMR)**:
- **Relevance**: Matches the specific request of the user.
- **Diversity**: Ensures each recommendation provides a unique thematic or artistic perspective compared to other results in the set.
- **Result**: Delivers a balanced selection of direct matches, thematic parallels, and conceptually linked discoveries.

---

## 4. Context-Aware Explanations

The explanation engine provides grounded justifications for each recommendation based on a prioritized hierarchy of data signals:
- **Conceptual Grounding**: Justifications based on high-level thematic clusters.
- **Personnel Grounding**: Connections identified through shared directors or leading actors.
- **Thematic Grounding**: Specific keyword and plot device overlaps.

---

This framework ensures that MovieMind remains both technically rigorous and intuitively accessible for high-precision movie discovery.
