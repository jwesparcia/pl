"""
export_tfjs.py
==============================================================================
Converts the trained PyTorch collaborative-filtering model into a format that
TensorFlow.js can load in the browser via tf.loadLayersModel().

No TensorFlow installation needed — we write the TF.js LayersModel JSON + raw
float32 binary directly from PyTorch tensors.

Output files (written to frontend/):
  model/model.json            — TF.js LayersModel descriptor
  model/group1-shard1of1.bin  — raw float32 embedding weights
  model/movie_biases.json     — per-movie bias values [num_movies]
  movies.json                 — [{movieId, movieIdx, title, genres}, ...]
==============================================================================
"""

import os
import json
import struct
import pickle
import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(ROOT, "..", "model")
FRONT_DIR   = os.path.join(ROOT, "..", "frontend")
OUT_MODEL   = os.path.join(FRONT_DIR, "model")

os.makedirs(OUT_MODEL, exist_ok=True)

# ── Load PyTorch checkpoint ───────────────────────────────────────────────────
print("Loading PyTorch checkpoint ...")
ckpt = torch.load(os.path.join(MODEL_DIR, "recommender.pt"),
                  map_location="cpu", weights_only=False)

num_users  = ckpt["num_users"]
num_movies = ckpt["num_movies"]
embed_dim  = ckpt["embed_dim"]
sd         = ckpt["state_dict"]

print(f"  num_users={num_users}  num_movies={num_movies}  embed_dim={embed_dim}")

# Extract weight arrays (float32)
movie_emb_w   = sd["movie_embedding.weight"].numpy().astype(np.float32)  # [N, D]
movie_bias_w  = sd["movie_bias.weight"].numpy().astype(np.float32)       # [N, 1]

print(f"  movie_embedding shape : {movie_emb_w.shape}")
print(f"  movie_bias shape      : {movie_bias_w.shape}")

# ── Write TF.js LayersModel ───────────────────────────────────────────────────
# The binary shard contains ONLY the movie embedding weights (what the browser
# needs for similarity + prediction).  Biases are stored separately as JSON
# because they are tiny and simpler to access from JavaScript.

BIN_FILENAME = "group1-shard1of1.bin"
bin_bytes    = movie_emb_w.tobytes()   # little-endian float32, shape [N, D]

with open(os.path.join(OUT_MODEL, BIN_FILENAME), "wb") as f:
    f.write(bin_bytes)
print(f"  Wrote {BIN_FILENAME}  ({len(bin_bytes):,} bytes)")

# model.json — TF.js Sequential LayersModel descriptor
model_json = {
    "format": "layers-model",
    "generatedBy": "keras v2.12.0",
    "convertedBy": "exported by export_tfjs.py (PyTorch -> TF.js)",
    "modelTopology": {
        "class_name": "Sequential",
        "config": {
            "name": "movie_recommender",
            "layers": [
                {
                    "class_name": "Embedding",
                    "config": {
                        "name": "movie_embedding",
                        "trainable": False,
                        "dtype": "float32",
                        "batch_input_shape": [None, 1],
                        "input_dim": int(num_movies),
                        "output_dim": int(embed_dim),
                        "embeddings_initializer": {
                            "class_name": "Zeros",
                            "config": {}
                        },
                        "embeddings_regularizer": None,
                        "activity_regularizer": None,
                        "embeddings_constraint": None,
                        "mask_zero": False,
                        "input_length": 1
                    }
                }
            ]
        }
    },
    "weightsManifest": [
        {
            "paths": [BIN_FILENAME],
            "weights": [
                {
                    "name": "movie_embedding/embeddings",
                    "shape": [int(num_movies), int(embed_dim)],
                    "dtype": "float32"
                }
            ]
        }
    ]
}

with open(os.path.join(OUT_MODEL, "model.json"), "w") as f:
    json.dump(model_json, f, indent=2)
print("  Wrote model.json")

# ── Write movie biases JSON ───────────────────────────────────────────────────
# Flatten from [N, 1] → [N]
biases_flat = movie_bias_w.squeeze(1).tolist()
with open(os.path.join(OUT_MODEL, "movie_biases.json"), "w") as f:
    json.dump(biases_flat, f)
print(f"  Wrote movie_biases.json  ({len(biases_flat)} values)")

# ── Load metadata & write movies.json ────────────────────────────────────────
print("\nLoading metadata.pkl ...")
with open(os.path.join(MODEL_DIR, "metadata.pkl"), "rb") as f:
    meta = pickle.load(f)

movie2idx  = meta["movie2idx"]   # {movieId: embeddingRowIndex}
idx2movie  = meta["idx2movie"]   # {embeddingRowIndex: movieId}
movies_df  = meta["movies_df"]   # DataFrame: movieId, title, genres

# Build a lookup: movieId → (title, genres)
movie_info = {}
for _, row in movies_df.iterrows():
    movie_info[int(row["movieId"])] = {
        "title":  str(row["title"]),
        "genres": str(row["genres"]),
    }

# Build movies list ordered by embedding row index (idx)
movies_list = []
for idx in range(num_movies):
    mid = idx2movie.get(idx)
    if mid is None:
        continue
    mid = int(mid)
    info = movie_info.get(mid, {"title": f"Movie {mid}", "genres": "(no genres listed)"})
    movies_list.append({
        "movieId":  mid,
        "movieIdx": idx,          # row index into embedding matrix
        "title":    info["title"],
        "genres":   info["genres"],
    })

movies_out = os.path.join(FRONT_DIR, "movies.json")
with open(movies_out, "w", encoding="utf-8") as f:
    json.dump(movies_list, f, ensure_ascii=False)
print(f"  Wrote movies.json  ({len(movies_list)} movies)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n✅  Export complete!")
print(f"   frontend/model/model.json")
print(f"   frontend/model/{BIN_FILENAME}")
print(f"   frontend/model/movie_biases.json")
print(f"   frontend/movies.json")
print("\nServe the frontend with:")
print("   cd e:\\moviereco\\frontend")
print("   python -m http.server 8080")
print("   Open: http://localhost:8080")
