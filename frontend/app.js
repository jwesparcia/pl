/**
 * app.js — MovieMind Frontend Logic
 * ===========================================================
 * Connects the HTML UI to the Flask REST API.
 *
 * Flow:
 *   1. On load  → fetch all movies for browse grid + genre chips
 *   2. Mode A   → "Pick a Movie" (cold start / dummy user)
 *                  User types a title → autocomplete dropdown
 *                  Select → POST /recommend-by-movie → cards
 *   3. Mode B   → "User ID" (existing user)
 *                  Enter ID → POST /recommend → cards
 *   4. On card  → open detail modal
 * ===========================================================
 */

/* ── Config ─────────────────────────────────────────────── */
const API_BASE = "http://localhost:5000";  // Flask server URL
const BROWSE_PAGE_SIZE = 20;               // Movies shown per "Load more"

/* ── State ──────────────────────────────────────────────── */
let allMovies       = [];   // All movies from /movies
let filteredMovies  = [];   // After search / genre filter
let browseOffset    = 0;    // Pagination cursor for browse grid
let activeGenre     = "All";
let lastRecommendations = [];

/* ── DOM refs ─────────────────────────────────────────── */
// Shared
const spinner        = document.getElementById("spinner");
const statusText     = document.getElementById("status-text");
const movieGrid      = document.getElementById("movie-grid");
const resultsHeader  = document.getElementById("results-header");
const resultsTitle   = document.getElementById("results-title");
const resultsCount   = document.getElementById("results-count");
const browseGrid     = document.getElementById("browse-grid");
const chipList       = document.getElementById("chip-list");
const movieSearch    = document.getElementById("movie-search");
const loadMoreBtn    = document.getElementById("load-more-btn");
const modalTemplate  = document.getElementById("modal-template");
const aiBadge        = document.getElementById("ai-badge");
let   activeModal    = null;

// Mode toggle
const tabMovie       = document.getElementById("tab-movie");
const tabUser        = document.getElementById("tab-user");
const panelMovie     = document.getElementById("panel-movie");
const panelUser      = document.getElementById("panel-user");

// Cold start (movie mode)
const movieTitleInput  = document.getElementById("movie-title-input");
const recommendMovieBtn = document.getElementById("recommend-movie-btn");
const autocompleteList = document.getElementById("autocomplete-list");
const becauseBanner    = document.getElementById("because-banner");
const becauseTitle     = document.getElementById("because-title");

// User ID mode
const userInput      = document.getElementById("user-id-input");
const recommendBtn   = document.getElementById("recommend-btn");

/* ══════════════════════════════════════════════════════════
   STARTUP — fetch all movies
══════════════════════════════════════════════════════════ */
async function loadAllMovies() {
  try {
    const res = await fetch(`${API_BASE}/movies`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allMovies      = await res.json();
    filteredMovies = [...allMovies];
    populateGenreChips();
    renderBrowseGrid(true);
  } catch (err) {
    console.error("Could not load movies:", err);
    setStatus("⚠️  Could not reach the API. Is Flask running on port 5000?", "error");
  }
}

/* ── Genre chips ────────────────────────────────────────── */
function populateGenreChips() {
  // Gather every unique genre across all movies
  const genreSet = new Set();
  allMovies.forEach(m => {
    m.genres.split("|").forEach(g => {
      if (g && g !== "(no genres listed)") genreSet.add(g);
    });
  });

  // Sort alphabetically
  const sorted = [...genreSet].sort();
  sorted.forEach(genre => {
    const btn = document.createElement("button");
    btn.className   = "chip";
    btn.textContent = genre;
    btn.dataset.genre = genre;
    btn.addEventListener("click", () => setGenreFilter(genre, btn));
    chipList.appendChild(btn);
  });
}

function setGenreFilter(genre, btn) {
  // Deactivate all chips, activate selected
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
  btn.classList.add("active");
  activeGenre = genre;

  // Re-filter
  applyFilters();
}

/* ── Search + filter logic ──────────────────────────────── */
function applyFilters() {
  const query = movieSearch.value.toLowerCase().trim();

  filteredMovies = allMovies.filter(m => {
    const matchesGenre = activeGenre === "All" || m.genres.includes(activeGenre);
    const matchesSearch = !query ||
      m.title.toLowerCase().includes(query) ||
      m.genres.toLowerCase().includes(query);
    return matchesGenre && matchesSearch;
  });

  browseOffset = 0;
  renderBrowseGrid(true);
}

/* ═══════════════════════════════════════════════════════════
   RENDER — Browse Grid
═══════════════════════════════════════════════════════════ */
function renderBrowseGrid(reset = false) {
  if (reset) browseGrid.innerHTML = "";

  const page = filteredMovies.slice(browseOffset, browseOffset + BROWSE_PAGE_SIZE);
  browseOffset += page.length;

  page.forEach((movie, i) => {
    const card = createMovieCard(movie, null, i);
    browseGrid.appendChild(card);
  });

  // Hide "Load more" if nothing left
  loadMoreBtn.hidden = browseOffset >= filteredMovies.length;
}

/* ═══════════════════════════════════════════════════════════
   MODE TOGGLE
═══════════════════════════════════════════════════════════ */
function switchMode(mode) {
  // Toggle tabs
  tabMovie.classList.toggle("active", mode === "movie");
  tabUser.classList.toggle("active", mode === "user");

  // Toggle panels
  panelMovie.classList.toggle("active", mode === "movie");
  panelUser.classList.toggle("active", mode === "user");

  // Update default status text
  if (mode === "movie") {
    setStatus("Pick a movie you love, and we'll find your next favorite.");
  } else {
    setStatus('Enter a User ID and hit <strong>Get Recommendations</strong> to begin.');
  }
}

tabMovie.addEventListener("click", () => switchMode("movie"));
tabUser.addEventListener("click", () => switchMode("user"));

/* ═══════════════════════════════════════════════════════════
   AUTOCOMPLETE (movie title search)
═══════════════════════════════════════════════════════════ */
let acHighlight = -1; // Keyboard navigation index

function showAutocomplete(query) {
  autocompleteList.innerHTML = "";
  acHighlight = -1;

  if (!query || query.length < 2) {
    autocompleteList.hidden = true;
    return;
  }

  const q = query.toLowerCase();
  const matches = allMovies.filter(m =>
    m.title.toLowerCase().includes(q)
  ).slice(0, 8);

  if (!matches.length) {
    autocompleteList.hidden = true;
    return;
  }

  matches.forEach((movie, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;

    const titleSpan = document.createElement("span");
    titleSpan.className = "ac-title";
    titleSpan.textContent = movie.title;

    const genreSpan = document.createElement("span");
    genreSpan.className = "ac-genre";
    genreSpan.textContent = movie.genres.split("|").slice(0, 2).join(", ");

    li.appendChild(titleSpan);
    li.appendChild(genreSpan);

    li.addEventListener("click", () => selectAutocomplete(movie.title));
    autocompleteList.appendChild(li);
  });

  autocompleteList.hidden = false;
}

function selectAutocomplete(title) {
  movieTitleInput.value = title;
  autocompleteList.hidden = true;
  acHighlight = -1;
  getRecommendationsByMovie(title);
}

// Keyboard navigation (arrow keys + Enter)
movieTitleInput.addEventListener("keydown", e => {
  const items = autocompleteList.querySelectorAll("li");
  if (!items.length || autocompleteList.hidden) {
    if (e.key === "Enter") {
      e.preventDefault();
      getRecommendationsByMovie(movieTitleInput.value);
    }
    return;
  }

  if (e.key === "ArrowDown") {
    e.preventDefault();
    acHighlight = Math.min(acHighlight + 1, items.length - 1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    acHighlight = Math.max(acHighlight - 1, 0);
  } else if (e.key === "Enter") {
    e.preventDefault();
    if (acHighlight >= 0) {
      selectAutocomplete(items[acHighlight].querySelector(".ac-title").textContent);
    } else {
      getRecommendationsByMovie(movieTitleInput.value);
    }
    return;
  } else if (e.key === "Escape") {
    autocompleteList.hidden = true;
    return;
  } else {
    return;
  }

  items.forEach((item, i) => item.classList.toggle("highlighted", i === acHighlight));
});

movieTitleInput.addEventListener("input", debounce(() => {
  showAutocomplete(movieTitleInput.value);
}, 150));

// Close dropdown when clicking outside
document.addEventListener("click", e => {
  if (!e.target.closest(".autocomplete-wrapper")) {
    autocompleteList.hidden = true;
  }
});

/* ═══════════════════════════════════════════════════════════
   COLD START: Recommend by Movie Title
   (The "Dummy User" approach)
═══════════════════════════════════════════════════════════ */
async function getRecommendationsByMovie(title) {
  if (!title || !title.trim()) {
    setStatus("Please type a movie title first.", "error");
    return;
  }

  setLoading(true);
  setStatus(`AI is analyzing connections for "${title}"...`);
  movieGrid.innerHTML = "";
  resultsHeader.hidden = true;
  becauseBanner.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/recommend-by-movie`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ movie_title: title }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.description || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const { source_movie, recommendations, ai_status } = data;
    lastRecommendations = recommendations;

    if (!recommendations.length) {
      setStatus("No similar movies found.", "error");
      return;
    }

    // Show AI Provider badge
    if (aiBadge) {
      aiBadge.textContent = `Powered by ${ai_status || "Templates"}`;
      aiBadge.hidden = false;
    }

    // Show "Because you liked" banner
    becauseTitle.textContent = source_movie.title;
    becauseBanner.hidden = false;

    setStatus(`Found ${recommendations.length} movies similar to "${source_movie.title}"`);
    resultsTitle.textContent = "Recommended for You";
    resultsHeader.hidden = false;
    resultsCount.textContent = `${recommendations.length} movies`;

    recommendations.forEach((movie, idx) => {
      const card = createMovieCard(movie, movie.predicted_rating, idx);
      card.style.animationDelay = `${idx * 50}ms`;
      movieGrid.appendChild(card);
    });

    document.getElementById("results-section").scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    console.error(err);
    setStatus(`${err.message || "Something went wrong. Is Flask running?"}`, "error");
  } finally {
    setLoading(false);
  }
}

/* ═══════════════════════════════════════════════════════════
   RECOMMEND — call /recommend endpoint (User ID mode)
═══════════════════════════════════════════════════════════ */
async function getRecommendations() {
  const userId = parseInt(userInput.value, 10);
  if (isNaN(userId) || userId < 1) {
    setStatus("⚠️  Please enter a valid User ID (1–610).", "error");
    return;
  }

  setLoading(true);
  setStatus(`AI is re-ranking top picks for User ${userId}...`);
  movieGrid.innerHTML = "";
  resultsHeader.hidden = true;
  becauseBanner.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ user_id: userId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.description || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const { recommendations, ai_status } = data;
    lastRecommendations = recommendations;

    if (!recommendations.length) {
      setStatus("⚠️  No recommendations found for this user.", "error");
      return;
    }

    // Show AI Provider badge
    if (aiBadge) {
      aiBadge.textContent = `Powered by ${ai_status || "Templates"}`;
      aiBadge.hidden = false;
    }

    setStatus(`Top ${recommendations.length} picks for User ${userId}`);
    resultsTitle.textContent = "Top Picks for You";
    resultsHeader.hidden = false;
    resultsCount.textContent = `${recommendations.length} movies`;

    recommendations.forEach((movie, idx) => {
      // Stagger card animation
      const card = createMovieCard(movie, movie.predicted_rating, idx);
      card.style.animationDelay = `${idx * 50}ms`;
      movieGrid.appendChild(card);
    });

    // Scroll to results
    document.getElementById("results-section").scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    console.error(err);
    setStatus(`❌  ${err.message || "Something went wrong. Is Flask running?"}`, "error");
  } finally {
    setLoading(false);
  }
}

/* ═══════════════════════════════════════════════════════════
   BUILD — Movie Card element
=══════════════════════════════════════════════════════════ */
/**
 * createMovieCard(movie, predictedRating, index)
 *
 * @param {Object} movie            — { id, title, genres }
 * @param {number|null} predictedRating — if null, show as browse card
 * @param {number} index            — for animation stagger
 */
function createMovieCard(movie, predictedRating, index) {
  const card = document.createElement("article");
  card.className = "movie-card";
  card.tabIndex  = 0;
  card.setAttribute("aria-label", movie.title);

  // Assign a deterministic accent color per genre
  const accentColor = genreColor(movie.genres);
  const colorBar    = document.createElement("div");
  colorBar.className           = "card-color-bar";
  colorBar.style.background    = accentColor;

  const body = document.createElement("div");
  body.className = "card-body";

  // Rank badge (only for recommendations)
  if (predictedRating !== null && movie.rank) {
    const rank = document.createElement("div");
    rank.className   = "card-rank";
    rank.textContent = `#${movie.rank} Pick`;
    body.appendChild(rank);
  }

  // Title
  const title = document.createElement("h3");
  title.className   = "card-title";
  title.textContent = movie.title;
  body.appendChild(title);

  // Genre badges
  const genreWrap = document.createElement("div");
  genreWrap.className = "card-genres";
  movie.genres.split("|").slice(0, 3).forEach(g => {
    if (g && g !== "(no genres listed)") {
      const badge = document.createElement("span");
      badge.className   = "genre-badge";
      badge.textContent = g;
      genreWrap.appendChild(badge);
    }
  });
  body.appendChild(genreWrap);

  // Star rating (recommendations only)
  if (predictedRating !== null) {
    const ratingWrap = document.createElement("div");
    ratingWrap.className = "card-rating";

    const stars = document.createElement("span");
    stars.className   = "stars";
    stars.textContent = starsForRating(predictedRating);

    const num = document.createElement("span");
    num.className   = "rating-num";
    num.textContent = `${predictedRating.toFixed(1)} / 5`;

    ratingWrap.appendChild(stars);
    ratingWrap.appendChild(num);
    body.appendChild(ratingWrap);
  }

  // Reason text (only populated for rule-based engine in cold-start mode)
  if (movie.reason) {
    const reasonPara = document.createElement("p");
    reasonPara.className = "card-reason";
    reasonPara.innerHTML = `<strong>Reason:</strong> ${movie.reason}`;
    
    // Quick inline styling for the reason
    reasonPara.style.fontSize = "0.85rem";
    reasonPara.style.color = "var(--text-dim)";
    reasonPara.style.marginTop = "0.75rem";
    reasonPara.style.lineHeight = "1.4";
    reasonPara.style.fontStyle = "italic";

    body.appendChild(reasonPara);
  }

  card.appendChild(colorBar);
  card.appendChild(body);

  // Open modal on click or Enter key
  const openModal = () => showModal(movie, predictedRating);
  card.addEventListener("click", openModal);
  card.addEventListener("keydown", e => { if (e.key === "Enter") openModal(); });

  return card;
}

/* ═══════════════════════════════════════════════════════════
   MODAL
═══════════════════════════════════════════════════════════ */
function showModal(movie, predictedRating) {
  // 1. Create from template
  const clone = modalTemplate.content.cloneNode(true);
  activeModal = clone.querySelector(".modal-overlay");

  const title  = activeModal.querySelector("#modal-title");
  const genres = activeModal.querySelector("#modal-genres");
  const rating = activeModal.querySelector("#modal-rating");
  const close  = activeModal.querySelector("#modal-close");

  title.textContent = movie.title;

  // 2. Genres
  genres.innerHTML = "";
  movie.genres.split("|").forEach(g => {
    if (g && g !== "(no genres listed)") {
      const badge = document.createElement("span");
      badge.className   = "genre-badge";
      badge.textContent = g;
      genres.appendChild(badge);
    }
  });

  // 3. Rating
  if (predictedRating !== null) {
    rating.textContent = `${starsForRating(predictedRating)}  ${predictedRating.toFixed(2)} / 5.0`;
    rating.hidden      = false;
  } else {
    rating.hidden = true;
  }

  // 4. Events for this instance
  close.addEventListener("click", closeModal);
  activeModal.addEventListener("click", e => { if (e.target === activeModal) closeModal(); });

  // 5. Append and Show
  document.body.appendChild(activeModal);
  
  // Force reflow for animation
  activeModal.offsetHeight; 
  activeModal.classList.add("active");
  activeModal.style.display = "flex";
  
  close.focus();
}

function closeModal() {
  if (!activeModal) return;
  activeModal.classList.remove("active");
  activeModal.style.display = "none";
  activeModal.remove(); // Remove from DOM entirely
  activeModal = null;
}

document.addEventListener("keydown", e => { 
  if (e.key === "Escape") closeModal(); 
});

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
/**
 * starsForRating — convert a 0-5 numeric rating to ★★★★☆ style string.
 * Full stars for every point, half star for 0.3–0.7 remainder.
 */
function starsForRating(rating) {
  const full = Math.floor(rating);
  const half = (rating - full) >= 0.3 ? 1 : 0;
  const empty = 5 - full - half;
  return "★".repeat(full) + (half ? "½" : "") + "☆".repeat(empty);
}

/**
 * genreColor — deterministic gradient per primary genre for card bar accent.
 * Keeps the grid visually varied without random values (stable across renders).
 */
const GENRE_COLORS = {
  "Action":     "linear-gradient(90deg,#fc5c7d,#6a3093)",
  "Adventure":  "linear-gradient(90deg,#f7971e,#ffd200)",
  "Animation":  "linear-gradient(90deg,#43e97b,#38f9d7)",
  "Comedy":     "linear-gradient(90deg,#ffecd2,#fcb69f)",
  "Crime":      "linear-gradient(90deg,#4b6cb7,#182848)",
  "Drama":      "linear-gradient(90deg,#a18cd1,#fbc2eb)",
  "Fantasy":    "linear-gradient(90deg,#a1c4fd,#c2e9fb)",
  "Horror":     "linear-gradient(90deg,#434343,#000000)",
  "Romance":    "linear-gradient(90deg,#f953c6,#b91d73)",
  "Sci-Fi":     "linear-gradient(90deg,#00c6ff,#0072ff)",
  "Thriller":   "linear-gradient(90deg,#373b44,#4286f4)",
  "Western":    "linear-gradient(90deg,#c94b4b,#4b134f)",
  "Musical":    "linear-gradient(90deg,#f7ff00,#db36a4)",
  "Documentary":"linear-gradient(90deg,#3a7bd5,#3a6073)",
};

function genreColor(genres) {
  const firstGenre = genres.split("|")[0];
  return GENRE_COLORS[firstGenre] || "linear-gradient(90deg,#7c5cfc,#00d4b4)";
}

/** Set the status message below search panel. */
function setStatus(msg, type = "info") {
  statusText.innerHTML = msg;
  if (type === "error") {
    // Wrap in error box if it's not inside one already
    if (!statusText.closest(".error-box")) {
      statusText.className = "status-text";
      statusText.style.color = "var(--accent-pink)";
    }
  } else {
    statusText.style.color = "";
  }
}

/** Toggle loading state (spinner + button disabled). */
function setLoading(on) {
  spinner.hidden      = !on;
  recommendBtn.disabled = on;
  if (!on && spinner.hidden) {
    // no-op, spinner already hidden
  }
}

/* ═══════════════════════════════════════════════════════════
   EVENT LISTENERS
═══════════════════════════════════════════════════════════ */
recommendBtn.addEventListener("click", getRecommendations);
recommendMovieBtn.addEventListener("click", () => {
  getRecommendationsByMovie(movieTitleInput.value);
});

userInput.addEventListener("keydown", e => {
  if (e.key === "Enter") getRecommendations();
});

loadMoreBtn.addEventListener("click", () => renderBrowseGrid(false));

movieSearch.addEventListener("input", debounce(applyFilters, 250));

// First chip ("All") is clicked by default — wire up its event
chipList.querySelector(".chip[data-genre='All']").addEventListener("click", function () {
  setGenreFilter("All", this);
});

/* Utility: debounce search input so we don't re-render on every keystroke */
function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

/* ── Bootstrap ──────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  loadAllMovies();
});
