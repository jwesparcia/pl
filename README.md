# MovieMind - Hybrid Intelligent Recommendation System

MovieMind is a sophisticated movie recommendation engine that integrates Semantic Vector Search (utilizing MiniLM Transformers) with Weighted Thematic Analysis and Neural Collaborative Filtering to deliver precise and context-aware suggestions.

Unlike traditional systems that rely solely on genre matching, MovieMind analyzes the underlying thematic concepts of cinematographic works to identify deep narrative connections.

---

## Project Structure

```
moviereco/
|-- backend/
|   |-- app.py                   # Main Flask REST API (Hybrid Engine)
|   |-- build_tmdb_metadata.py    # Data pipeline: metadata extraction and vectorization
|   |-- train_model.py           # Neural training: Keras/Torch collaborative filtering
|   |-- reco_utils.py            # Intelligent Explanation and Utility layer
|   `-- requirements.txt         # System dependencies
|-- frontend/
|   |-- index.html               # Glassmorphism User Interface
|   |-- style.css                # Interface styling
|   `-- app.js                   # Frontend client logic
|-- README.md                    # Primary documentation
`-- recommendation_logic.md      # Detailed algorithmic overview
```

---

## Quick Start

### Step 1 - Initialize Environment
Create a virtual environment and install the required dependencies:

```bash
# From the root directory
python -m venv venv
.\venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Step 2 - Generate Search Index
Execute the metadata build script to process the TMDB 5000 dataset and generate 384-dimensional semantic embeddings.

```bash
cd backend
python build_tmdb_metadata.py
```
This process generates `model/movies.json`, which serves as the primary search index.

### Step 3 - Train Neural Recommender
Execute the training script to generate the Keras collaborative filtering model used for final ranking.

```bash
python train_model.py
```
This generates `model/recommender.keras` and the associated metadata.

### Step 4 - Start the Application
Launch the Flask API on port 5005:

```bash
python app.py
```

### Step 5 - Access the Interface
Open `frontend/index.html` in a web browser. It is recommended to use a local development server for the best experience.

---

## Deployment (Render)

MovieMind is optimized for deployment on the Render platform:

1.  **Repository Configuration**: Ensure `backend/requirements.txt` and `render.yaml` are present in the repository.
2.  **Blueprint Connection**: Create a new Blueprint service on Render and connect the GitHub repository.
3.  **Resource Management**: The system is configured to run efficiently within Render's free tier (512MB RAM).

---

## Technical Stack

| Component | Technology |
| :--- | :--- |
| **Neural Framework** | Keras 3 (PyTorch Backend) |
| **Semantic Intelligence** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Data Processing** | Scikit-learn, Pandas, NumPy |
| **Backend API** | Flask, Flask-CORS |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Dataset** | TMDB 5000 Movies |

---

## System Requirements

- Python 3.14+
- 8GB RAM recommended for optimal model loading performance.
- Active internet connection for the initial download of Transformer weights.

---
Developed as a demonstration of next-generation recommendation systems.