#  MovieMind — Hybrid Intelligent Recommendation System

MovieMind is a modern, high-intelligence movie recommendation engine that balances **Semantic Vector Search** (using MiniLM Transformers) with **Weighted Thematic Analysis** to deliver precise, diverse, and context-aware suggestions.

Unlike traditional systems that only match genres, MovieMind understands the *meaning* and *themes* behind a movie (e.g., recognizing that "Artificial Intelligence" and "Cyborgs" are related).

---

##  Project Structure

```
moviereco/
├── backend/
│   ├── app.py                   # Main Flask REST API (Hybrid Engine)
│   ├── build_tmdb_metadata.py    # Data pipeline: extraction & vectorization
│   ├── reco_utils.py            # Intelligent Explanation & Utility layer
│   └── requirements.txt         # Core dependencies
├── frontend/
│   ├── index.html               # Premium Glassmorphism UI
│   ├── style.css                # Dark-mode styling
│   └── app.js                   # Frontend client logic
├── README.md                    # You are here
└── recommendation_logic.md      # Deep dive into the math & algorithms
```

---

## 🚀 Quick Start

### Step 1 — Initialize Environment
Create a virtual environment and install the required AI and backend libraries:

```bash
# From root directory
python -m venv venv
.\venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Step 2 — Build Metadata & Search Index
This script processes the **TMDB 5000 dataset**, extracts high-value themes, and generates 384-dimensional semantic embeddings.

```bash
cd backend
python build_tmdb_metadata.py
```
*Note: This generates `model/movies.json` which acts as our high-dimensional search index.*
To ensure the model is ready for deployment, make sure `model/movies.json` exists before pushing to GitHub.

### Step 3 — Start the Hybrid Engine
Launch the Flask API on the optimized port (**5005**):

```bash
python app.py
```

### Step 4 — Launch the UI
Simply open `frontend/index.html` in your browser. For the best experience, use a local server like VS Code's **Live Server**.

---

| **Concept Explanations** | Generates grounded, human-like reasons for every recommendation. |

---

## ☁️ Deployment (Render)

MovieMind is ready for one-click deployment on **Render's Free Tier**:

1.  **Push to GitHub**: Ensure your `backend/requirements.txt` and `render.yaml` are committed.
2.  **Connect Repo**: Log in to Render, click **New +**, and select **Blueprint**.
3.  **Deploy**: Connect your GitHub repository. Render will automatically detect the `render.yaml` and configure the service.
4.  **Scaling**: The free tier provides 512MB RAM, which is sufficient for the `all-MiniLM-L6-v2` model and ~5000 movies.

*Note: Initial deployment may take 3-5 minutes as Render builds the environment and installs AI dependencies.*

---

## 🛠 Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Semantic AI** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Data Science** | `scikit-learn`, `pandas`, `numpy` |
| **Backend** | `Flask`, `Flask-CORS` |
| **Frontend** | HTML5, Vanilla CSS, Vanilla JS |
| **Dataset** | TMDB 5000 Movies |

---

## 📋 Requirements

- Python 3.10+
- 8GB+ RAM (recommended for loading semantic models)
- Internet connection (initial run only to download the Transformer model)

---
*Built with ❤️ as a Next-Gen Recommendation System Demonstration.*