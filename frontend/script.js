/**
 * script.js — MovieMind TF.js Recommendation Engine
 * ============================================================
 * Runs entirely in the browser. No backend, no database.
 *
 * HOW COLLABORATIVE FILTERING WORKS (quick primer)
 * --------------------------------------------------
 * During training, every movie was assigned a learned dense
 * vector (embedding) that captures its "taste fingerprint".
 * Movies that tend to be enjoyed by the same people land near
 * each other in this latent space — regardless of genre.
 *
 * HOW THIS ENGINE WORKS
 * ----------------------
 * 1. Load model.json (TF.js LayersModel) → extract the
 *    movie-embedding matrix  [num_movies × embed_dim]
 * 2. When user picks a movie:
 *    a) Get that movie's embedding vector
 *    b) Compute cosine similarity to ALL other movies
 *    c) Build a "dummy user" = weighted avg of selected + top-4 similar
 *    d) Predict each movie's rating: sigmoid(dot(user, movie) + bias)
 *    e) Hybrid score = 0.6 × predicted + 0.4 × cosine_similarity
 *    f) Filter already-liked movies, sort, show top 10
 *
 * COLD START PROBLEM
 * ------------------
 * We have no real user history. The "dummy user" trick solves this
 * by injecting a preference profile built purely from the selected
 * movie's embedding neighbourhood. Because the dummy user is computed
 * fresh every time, each movie selection yields a different result.
 * ============================================================
 */

/* ── Config ─────────────────────────────────────────────────── */
const MODEL_URL      = "model/model.json";   // TF.js LayersModel
const MOVIES_URL     = "movies.json";
const BIASES_URL     = "model/movie_biases.json";
const BROWSE_PAGE    = 20;                   // Cards per "Load more"
const TOP_K_SIMILAR  = 4;                    // Similar movies for dummy user
const TOP_N_RESULTS  = 10;                   // Final recommendations to show
const SIMILAR_WEIGHT = [2.0, 1.0, 0.8, 0.6, 0.4]; // Dummy-user profile weights
const SIM_THRESHOLD  = 0.10;                // Min cosine similarity to include
const W_PREDICTED    = 0.60;               // Hybrid weight for predicted rating
const W_SIMILARITY   = 0.40;               // Hybrid weight for cosine similarity

/* ── State ──────────────────────────────────────────────────── */
let model           = null;   // tf.LayersModel
let movieEmbeddings = null;   // tf.Tensor2D [N × D]
let movieBiases     = null;   // Float32Array [N]
let movies          = [];     // [{movieId, movieIdx, title, genres}]
let titleToIdx      = {};     // title → movies[] index
let allMovies       = [];     // alias for browse
let filteredMovies  = [];
let browseOffset    = 0;
let activeGenre     = "All";
let acHighlight     = -1;

/* ── DOM ─────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const modelStatusEl    = $("model-status");
const modelFillEl      = $("model-status-fill");
const modelLabelEl     = $("model-status-label");
const titleInput       = $("movie-title-input");
const recommendBtn     = $("recommend-btn");
const autocompleteList = $("autocomplete-list");
const spinnerEl        = $("spinner");
const statusText       = $("status-text");
const becauseBanner    = $("because-banner");
const becauseTitle     = $("because-title");
const similarHint      = $("similar-hint");
const similarMoviesEl  = $("similar-movies");
const resultsHeader    = $("results-header");
const resultsTitleEl   = $("results-title");
const resultsCount     = $("results-count");
const movieGrid        = $("movie-grid");
const browseGrid       = $("browse-grid");
const chipList         = $("chip-list");
const movieSearch      = $("movie-search");
const loadMoreBtn      = $("load-more-btn");
let activeModal        = null;

/* ══════════════════════════════════════════════════════════════
   INITIALISATION
══════════════════════════════════════════════════════════════ */
async function init() {
  setModelStatus("loading", "Loading AI model…", 10);
  try {
    // Parallel-fetch movies data and model
    const [moviesData, biasesData, tfModel] = await Promise.all([
      fetch(MOVIES_URL).then(r => r.json()),
      fetch(BIASES_URL).then(r => r.json()),
      tf.loadLayersModel(MODEL_URL),
    ]);

    setModelStatus("loading", "Extracting embeddings…", 70);

    model = tfModel;
    movies = moviesData;
    allMovies = [...movies];
    filteredMovies = [...movies];
    movieBiases = new Float32Array(biasesData);

    // Build title → array-index lookup
    movies.forEach((m, i) => { titleToIdx[m.title] = i; });

    // Extract movie embedding matrix [N × D] from the embedding layer.
    // This is the core weight tensor produced by collaborative filtering training.
    const embLayer = model.getLayer("movie_embedding");
    movieEmbeddings = embLayer.getWeights()[0]; // tf.Tensor2D

    console.log(
      `%c[MovieMind] Model loaded`,
      "color:#7c5cfc;font-weight:bold",
      `— ${movies.length} movies, embeddings shape: [${movieEmbeddings.shape}]`
    );

    setModelStatus("ready", `Model ready — ${movies.length} movies indexed ✓`, 100);

    // Enable UI
    titleInput.disabled = false;
    recommendBtn.disabled = false;
    movieSearch.disabled = false;

    populateGenreChips();
    renderBrowseGrid(true);
    setStatus("Select a movie above to get personalised recommendations.");

  } catch (err) {
    console.error("[MovieMind] Init error:", err);
    setModelStatus("error", `Failed to load model: ${err.message}`, 0);
    setStatus("⚠️ Could not load the AI model. See console for details.", true);
  }
}

function setModelStatus(state, label, pct) {
  modelFillEl.style.width = pct + "%";
  modelLabelEl.textContent = label;
  modelStatusEl.className = "model-status " + state;
}

/* ══════════════════════════════════════════════════════════════
   CORE RECOMMENDATION ENGINE
   --------------------------
   All tensor math runs in WebGL (via TF.js) — no server needed.
══════════════════════════════════════════════════════════════ */
async function getRecommendations(selectedTitle) {
  const selectedTitle_ = selectedTitle.trim();
  if (!selectedTitle_) { setStatus("Please enter a movie title.", true); return; }

  const selMovieIdx = movies.findIndex(m => m.title === selectedTitle_);
  if (selMovieIdx === -1) { setStatus(`Movie "${selectedTitle_}" not found. Try the autocomplete.`, true); return; }

  const selMovie    = movies[selMovieIdx];
  const selEmbIdx   = selMovie.movieIdx; // row in embedding matrix

  setLoading(true);
  setStatus(`Analysing embedding space for "${selMovie.title}"…`);
  movieGrid.innerHTML = "";
  resultsHeader.hidden = true;
  becauseBanner.hidden = true;

  console.groupCollapsed(`%c[SELECTED] ${selMovie.title} (movieIdx: ${selEmbIdx})`, "color:#00d4b4;font-weight:bold");

  try {
    // ── STEP 1: Extract selected movie embedding ─────────────────
    // Slice a single row [1 × D] from the matrix, squeeze to [D]
    const selectedEmb = tf.tidy(() =>
      movieEmbeddings.slice([selEmbIdx, 0], [1, -1]).squeeze()  // [D]
    );

    // ── STEP 2: Compute cosine similarity to ALL movies ──────────
    // Normalise every row & dot with selectedEmb → [N] similarity scores
    // Cosine similarity measures the angle between two vectors — it is
    // invariant to magnitude, so it purely captures directional alignment.
    const { similarities, simArray } = await tf.tidy(() => {
      const norms = tf.norm(movieEmbeddings, 2, 1, true).add(1e-8); // [N,1]
      const normEmbs = movieEmbeddings.div(norms);                   // [N,D]
      const selNorm  = selectedEmb.div(tf.norm(selectedEmb).add(1e-8)); // [D]
      const sims     = normEmbs.matMul(selNorm.expandDims(1)).squeeze(); // [N]
      return { similarities: sims, simArray: sims };
    });
    const simArr = await similarities.data(); // Float32Array [N]

    // ── STEP 3: Find top-K similar movies (build dummy user profile) ──
    // Sort by cosine similarity, exclude the selected movie itself.
    // These top-K movies become the "preference context" for the dummy user.
    const sortedBySimm = Array.from(simArr)
      .map((s, embIdx) => ({ embIdx, sim: s }))
      .filter(x => x.embIdx !== selEmbIdx)
      .sort((a, b) => b.sim - a.sim);

    const topKSimilar = sortedBySimm.slice(0, TOP_K_SIMILAR);

    // Reverse-lookup: embIdx → movie record
    const embIdxToMovie = {};
    movies.forEach(m => { embIdxToMovie[m.movieIdx] = m; });

    const topKMovies = topKSimilar.map(x => ({
      movie: embIdxToMovie[x.embIdx],
      embIdx: x.embIdx,
      sim: x.sim,
    }));

    console.log(
      "%c[SIMILAR] Top-5 neighbours by cosine similarity:",
      "color:#7c5cfc",
      sortedBySimm.slice(0, 5).map(x => {
        const m = embIdxToMovie[x.embIdx];
        return `${m?.title} (cos=${x.sim.toFixed(4)})`;
      })
    );

    // ── STEP 4: Build dummy user embedding ───────────────────────
    // The dummy user is the weighted centroid of the selected movie
    // embedding + its top-K neighbours. This captures a broader slice
    // of latent space, reducing the cold-start sensitivity.
    // Weight vector: [2.0, 1.0, 0.8, 0.6, 0.4] → selected gets 2×.
    const profileEmbIndices = [selEmbIdx, ...topKSimilar.map(x => x.embIdx)];
    const weights           = SIMILAR_WEIGHT.slice(0, profileEmbIndices.length);
    const weightSum         = weights.reduce((a, b) => a + b, 0);

    const dummyUserEmb = tf.tidy(() => {
      const stack = tf.stack(
        profileEmbIndices.map(i =>
          movieEmbeddings.slice([i, 0], [1, -1]).squeeze()  // [D]
        )
      );  // [K × D]
      const wTensor = tf.tensor1d(weights).reshape([-1, 1]);   // [K,1]
      return stack.mul(wTensor).sum(0).div(weightSum);          // [D]
    });

    // ── STEP 5: Predict rating for every movie ───────────────────
    // predicted = sigmoid( dot(dummyUser, movieEmb) + movieBias )
    // The dot product measures how well the movie aligns with the
    // dummy user's taste profile; sigmoid squashes to [0,1].
    const predictedScores = tf.tidy(() => {
      const dots   = movieEmbeddings.matMul(dummyUserEmb.expandDims(1)).squeeze(); // [N]
      const biases = tf.tensor1d(movieBiases);                                      // [N]
      return dots.add(biases).sigmoid();                                            // [N]
    });
    const predArr = await predictedScores.data(); // Float32Array [N]

    // ── STEP 6: Hybrid scoring ───────────────────────────────────
    // final_score = 0.6 × predicted_rating + 0.4 × cosine_similarity
    // Blending both signals ensures we avoid recommending just the
    // globally most predicted movies (popularity bias) and stay anchored
    // to the user's specific selection.
    const hybridScores = tf.tidy(() =>
      predictedScores.mul(W_PREDICTED).add(similarities.mul(W_SIMILARITY))
    );
    const hybridArr = await hybridScores.data(); // Float32Array [N]

    // ── STEP 7: Filter, sort, top-N ─────────────────────────────
    const likedEmbSet = new Set(profileEmbIndices);

    const candidates = movies
      .filter(m => {
        if (likedEmbSet.has(m.movieIdx)) return false;   // already "liked"
        if (simArr[m.movieIdx] < SIM_THRESHOLD) return false; // too distant
        return true;
      })
      .map(m => ({
        ...m,
        predicted:  predArr[m.movieIdx],
        similarity: simArr[m.movieIdx],
        score:      hybridArr[m.movieIdx],
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_N_RESULTS)
      .map((m, i) => ({ ...m, rank: i + 1 }));

    console.log(
      "%c[RESULTS] Top recommendations:",
      "color:#f953c6",
      candidates.map(m =>
        `#${m.rank} ${m.title} (predicted=${(m.predicted*5).toFixed(2)}, cos=${m.similarity.toFixed(4)}, hybrid=${m.score.toFixed(4)})`
      )
    );
    console.groupEnd();

    // ── STEP 8: Cleanup tensors ──────────────────────────────────
    tf.dispose([selectedEmb, similarities, dummyUserEmb, predictedScores, hybridScores]);

    // ── STEP 9: Render ───────────────────────────────────────────
    if (!candidates.length) {
      setStatus("No sufficiently similar movies found. Try a different title.", true);
      return;
    }

    // "Because you liked" banner
    becauseTitle.textContent = selMovie.title;
    becauseBanner.hidden = false;

    // "Similar to" hint
    if (topKMovies.length > 0) {
      const names = topKMovies.slice(0, 2).map(x => x.movie?.title).filter(Boolean);
      if (names.length) {
        similarMoviesEl.textContent = names.join(" · ");
        similarHint.hidden = false;
      }
    }

    resultsTitleEl.textContent = "Recommended for You";
    resultsHeader.hidden = false;
    resultsCount.textContent = `${candidates.length} movies`;

    candidates.forEach((movie, i) => {
      const card = createMovieCard(movie, i * 45);
      movieGrid.appendChild(card);
    });

    setStatus(`Found ${candidates.length} top picks based on "${selMovie.title}"`);
    $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    console.error("[MovieMind] Recommendation error:", err);
    console.groupEnd();
    setStatus("Something went wrong during recommendation. See console.", true);
  } finally {
    setLoading(false);
  }
}

/* ══════════════════════════════════════════════════════════════
   UI — AUTOCOMPLETE
══════════════════════════════════════════════════════════════ */
function showAutocomplete(query) {
  autocompleteList.innerHTML = "";
  acHighlight = -1;
  if (!query || query.length < 2) { autocompleteList.hidden = true; return; }

  const q = query.toLowerCase();
  const matches = movies.filter(m => m.title.toLowerCase().includes(q)).slice(0, 9);

  if (!matches.length) { autocompleteList.hidden = true; return; }

  matches.forEach((m, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", "false");

    const titleSpan = document.createElement("span");
    titleSpan.className = "ac-title";
    titleSpan.textContent = m.title;

    const genreSpan = document.createElement("span");
    genreSpan.className = "ac-genre";
    genreSpan.textContent = m.genres.split("|").slice(0, 2).join(", ");

    li.appendChild(titleSpan);
    li.appendChild(genreSpan);
    li.addEventListener("mousedown", e => {
      e.preventDefault();
      selectMovie(m.title);
    });
    autocompleteList.appendChild(li);
  });

  autocompleteList.hidden = false;
}

function selectMovie(title) {
  titleInput.value = title;
  autocompleteList.hidden = true;
  acHighlight = -1;
  getRecommendations(title);
}

titleInput.addEventListener("input", debounce(() => showAutocomplete(titleInput.value), 120));

titleInput.addEventListener("keydown", e => {
  const items = autocompleteList.querySelectorAll("li");
  if (!items.length || autocompleteList.hidden) {
    if (e.key === "Enter") { e.preventDefault(); getRecommendations(titleInput.value); }
    return;
  }
  if (e.key === "ArrowDown")  { e.preventDefault(); acHighlight = Math.min(acHighlight + 1, items.length - 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); acHighlight = Math.max(acHighlight - 1, 0); }
  else if (e.key === "Enter") {
    e.preventDefault();
    if (acHighlight >= 0) selectMovie(items[acHighlight].querySelector(".ac-title").textContent);
    else getRecommendations(titleInput.value);
    return;
  } else if (e.key === "Escape") { autocompleteList.hidden = true; return; }
  else return;

  items.forEach((el, i) => el.classList.toggle("highlighted", i === acHighlight));
});

document.addEventListener("mousedown", e => {
  if (!e.target.closest(".autocomplete-wrapper")) autocompleteList.hidden = true;
});

/* ══════════════════════════════════════════════════════════════
   UI — GENRE CHIPS + BROWSE GRID
══════════════════════════════════════════════════════════════ */
function populateGenreChips() {
  const genreSet = new Set();
  movies.forEach(m => m.genres.split("|").forEach(g => {
    if (g && g !== "(no genres listed)") genreSet.add(g);
  }));

  [...genreSet].sort().forEach(genre => {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.textContent = genre;
    btn.dataset.genre = genre;
    btn.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      activeGenre = genre;
      applyFilters();
    });
    chipList.appendChild(btn);
  });
}

function applyFilters() {
  const q = (movieSearch.value || "").toLowerCase().trim();
  filteredMovies = allMovies.filter(m => {
    const genreOk  = activeGenre === "All" || m.genres.includes(activeGenre);
    const searchOk = !q || m.title.toLowerCase().includes(q) || m.genres.toLowerCase().includes(q);
    return genreOk && searchOk;
  });
  browseOffset = 0;
  renderBrowseGrid(true);
}

function renderBrowseGrid(reset = false) {
  if (reset) browseGrid.innerHTML = "";
  const page = filteredMovies.slice(browseOffset, browseOffset + BROWSE_PAGE);
  browseOffset += page.length;
  page.forEach((m, i) => {
    const card = createMovieCard(m, i * 20, false);
    browseGrid.appendChild(card);
  });
  loadMoreBtn.hidden = browseOffset >= filteredMovies.length;
}

/* ══════════════════════════════════════════════════════════════
   UI — CARDS + MODAL
══════════════════════════════════════════════════════════════ */
/**
 * createMovieCard(movie, animDelay, isRecommendation)
 * movie: {movieId, movieIdx, title, genres, rank?, score?, predicted?, similarity?}
 * animDelay: ms delay for entry animation stagger
 * isRecommendation: if false, render as browse card
 */
function createMovieCard(movie, animDelay = 0, isRecommendation = true) {
  const card = document.createElement("article");
  card.className = "movie-card";
  card.tabIndex  = 0;
  card.setAttribute("aria-label", movie.title);
  card.style.animationDelay = `${animDelay}ms`;

  const colorBar = document.createElement("div");
  colorBar.className = "card-color-bar";
  colorBar.style.background = genreGradient(movie.genres);

  const body = document.createElement("div");
  body.className = "card-body";

  if (isRecommendation && movie.rank) {
    const rankEl = document.createElement("div");
    rankEl.className = "card-rank";
    rankEl.textContent = `#${movie.rank} Pick`;
    body.appendChild(rankEl);
  }

  const titleEl = document.createElement("h3");
  titleEl.className = "card-title";
  titleEl.textContent = movie.title;
  body.appendChild(titleEl);

  const genreWrap = document.createElement("div");
  genreWrap.className = "card-genres";
  movie.genres.split("|").slice(0, 3).forEach(g => {
    if (g && g !== "(no genres listed)") {
      const badge = document.createElement("span");
      badge.className = "genre-badge";
      badge.textContent = g;
      genreWrap.appendChild(badge);
    }
  });
  body.appendChild(genreWrap);

  // Score bar for recommendations
  if (isRecommendation && movie.score != null) {
    const scoreDiv = document.createElement("div");
    scoreDiv.className = "card-score";

    const label = document.createElement("div");
    label.className = "score-label";
    label.textContent = "Match score";

    const barWrap = document.createElement("div");
    barWrap.className = "score-bar-wrap";
    const barFill = document.createElement("div");
    barFill.className = "score-bar-fill";
    barFill.style.width = (Math.min(movie.score, 1) * 100).toFixed(1) + "%";
    barWrap.appendChild(barFill);

    const val = document.createElement("div");
    val.className = "score-value";
    val.textContent = `${(movie.score * 100).toFixed(1)}%  ·  est. ${(movie.predicted * 5).toFixed(1)} ★ / 5`;

    scoreDiv.appendChild(label);
    scoreDiv.appendChild(barWrap);
    scoreDiv.appendChild(val);
    body.appendChild(scoreDiv);
  }

  card.appendChild(colorBar);
  card.appendChild(body);

  const openFn = () => showModal(movie, isRecommendation);
  card.addEventListener("click", openFn);
  card.addEventListener("keydown", e => { if (e.key === "Enter") openFn(); });

  return card;
}

function showModal(movie, isRecommendation) {
  const tmpl  = $("modal-template");
  const clone = tmpl.content.cloneNode(true);
  activeModal = clone.querySelector(".modal-overlay");

  activeModal.querySelector("#modal-color-bar").style.background = genreGradient(movie.genres);
  activeModal.querySelector("#modal-title").textContent = movie.title;

  const genresEl = activeModal.querySelector("#modal-genres");
  movie.genres.split("|").forEach(g => {
    if (g && g !== "(no genres listed)") {
      const b = document.createElement("span");
      b.className = "genre-badge";
      b.textContent = g;
      genresEl.appendChild(b);
    }
  });

  const scoreRow = activeModal.querySelector("#modal-score-row");
  if (isRecommendation && movie.score != null) {
    scoreRow.innerHTML =
      `Hybrid score: <strong>${(movie.score * 100).toFixed(1)}%</strong> &nbsp;·&nbsp; ` +
      `Predicted rating: <strong>${(movie.predicted * 5).toFixed(2)} / 5</strong> &nbsp;·&nbsp; ` +
      `Cosine similarity: <strong>${(movie.similarity * 100).toFixed(1)}%</strong>`;
  } else {
    scoreRow.hidden = true;
  }

  const closeBtn = activeModal.querySelector("#modal-close");
  closeBtn.addEventListener("click", closeModal);
  activeModal.addEventListener("click", e => { if (e.target === activeModal) closeModal(); });

  document.body.appendChild(activeModal);
  requestAnimationFrame(() => {
    activeModal.classList.add("active");
    activeModal.style.display = "flex";
    closeBtn.focus();
  });
}

function closeModal() {
  if (!activeModal) return;
  activeModal.classList.remove("active");
  activeModal.style.display = "none";
  activeModal.remove();
  activeModal = null;
}

document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

/* ══════════════════════════════════════════════════════════════
   HELPERS
══════════════════════════════════════════════════════════════ */
const GENRE_GRADIENTS = {
  "Action":      "linear-gradient(90deg,#fc5c7d,#6a3093)",
  "Adventure":   "linear-gradient(90deg,#f7971e,#ffd200)",
  "Animation":   "linear-gradient(90deg,#43e97b,#38f9d7)",
  "Comedy":      "linear-gradient(90deg,#f093fb,#f5576c)",
  "Crime":       "linear-gradient(90deg,#4b6cb7,#182848)",
  "Documentary": "linear-gradient(90deg,#3a7bd5,#3a6073)",
  "Drama":       "linear-gradient(90deg,#a18cd1,#fbc2eb)",
  "Fantasy":     "linear-gradient(90deg,#a1c4fd,#c2e9fb)",
  "Horror":      "linear-gradient(90deg,#434343,#000000)",
  "Musical":     "linear-gradient(90deg,#f7ff00,#db36a4)",
  "Mystery":     "linear-gradient(90deg,#360033,#0b8793)",
  "Romance":     "linear-gradient(90deg,#f953c6,#b91d73)",
  "Sci-Fi":      "linear-gradient(90deg,#00c6ff,#0072ff)",
  "Thriller":    "linear-gradient(90deg,#373b44,#4286f4)",
  "Western":     "linear-gradient(90deg,#c94b4b,#4b134f)",
};

function genreGradient(genres) {
  const g = genres.split("|")[0];
  return GENRE_GRADIENTS[g] || "linear-gradient(90deg,#7c5cfc,#00d4b4)";
}

function setStatus(msg, isError = false) {
  statusText.textContent = msg;
  statusText.className = "status-text" + (isError ? " status-error" : "");
}

function setLoading(on) {
  spinnerEl.hidden  = !on;
  recommendBtn.disabled = on;
  titleInput.disabled   = on;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

/* ══════════════════════════════════════════════════════════════
   EVENT WIRING
══════════════════════════════════════════════════════════════ */
recommendBtn.addEventListener("click", () => getRecommendations(titleInput.value));
loadMoreBtn.addEventListener("click", () => renderBrowseGrid(false));
movieSearch.addEventListener("input", debounce(applyFilters, 200));

// "All" chip wired up immediately
chipList.querySelector(".chip[data-genre='All']").addEventListener("click", function () {
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
  this.classList.add("active");
  activeGenre = "All";
  applyFilters();
});

/* ── Bootstrap ──────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", init);
