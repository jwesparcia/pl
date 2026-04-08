# How MovieMind Computes Recommendations

MovieMind uses a **Content-Based Filtering** pipeline, relying on Term Frequency-Inverse Document Frequency (TF-IDF) and Cosine Similarity to mathematically find movies that share the most relevant features with your chosen title. 

Here is the step-by-step breakdown of how the engine works from startup to generating a recommendation:

## 1. Data Ingestion & Preprocessing
When the Flask server starts (`app.py`), the system extracts metadata from `movies.json`. It passes the list of all movies through the `preprocess_data()` function:
- **Title Normalization**: The release year is stripped from the title using regular expressions (e.g., `Toy Story (1995)` becomes `Toy Story`). This prevents the algorithm from overly obsessing over movies just because they were released in the same year.
- **Feature Concatenation**: The cleaned title and the list of genres (which are separated by `|` strings) are merged into one flat document string. 
  - *Example*: `Toy Story` + `Adventure|Animation|Children|Comedy|Fantasy` becomes the string document: `"Toy Story Adventure Animation Children Comedy Fantasy"`.

## 2. Building the TF-IDF Similarity Matrix
These formatted strings are passed into `build_similarity_matrix()`, which utilizes Scikit-Learn's `TfidfVectorizer`:
- **TF (Term Frequency)**: Counts how often a word appears in a specific movie's document.
- **IDF (Inverse Document Frequency)**: Lowers the weight of extremely common words (like "Drama" or "Comedy") and raises the weight of rare, highly-specific words (like "Animation", "Noir", or unique title keywords like "Matrix").
- **Matrix Generation**: It converts the entire ~9,700 movie database into an immense mathematical grid mathematically mapping the significance of every word/genre to every movie.

> **Why TF-IDF?**  
> It is extremely efficient. Rather than manually guessing which genres match or writing brittle `if/else` logic, TF-IDF inherently understands that if two movies share a rare genre/keyword, they are highly similar.

## 3. Querying the Engine (`recommend_movies()`)
When a user searches for a movie in the frontend:
1. **Title Matching**: The API tries to find the exact target movie ID in the dataset.
2. **Matrix Lookup**: It extracts the calculated TF-IDF vector array for that specific movie from our global matrix.
3. **Cosine Similarity**: Using `cosine_similarity`, the engine calculates the geometric angle (distance) between the queried movie's vector and *every single other movie vector* in our dataset. 
   - A score of `1.0` means identical content.
   - A score of `0.0` means they share absolutely zero overlapping terms.
4. **De-duplication**: The engine loops over the highest-scoring candidate indices. It asserts that the movie ID is not the source movie and not a movie that was already appended (avoiding identical exact matches dropping into the final array).

## 4. Explanations & Delivery
Because the AI/LLM generator was removed, the system procedurally translates the output.
Before sending the final Top-10 list back to the UI, the engine passes the candidates through `reco_utils.py`'s `get_movie_explanation` function. This matches the strongest genre overlaps between the source and target movie and picks from a dictionary of natural-language string templates, generating the dynamic explanations in real-time (e.g., *"If you enjoyed the Drama elements in X, you'll likely appreciate Y"*).
