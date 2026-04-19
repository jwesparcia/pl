/**
 * app.js - MovieMind Frontend Logic
 * ===========================================================
 * Connects the HTML UI to the Flask REST API.
 *
 * Flow:
 *   1. On load  -> fetch all movies for browse grid + genre chips
 *   2. Mode A   -> "Pick a Movie" (cold start / dummy user)
 *                  User types a title -> autocomplete dropdown
 *                  Select -> POST /recommend-by-movie -> cards
 *   3. Mode B   -> "User ID" (existing user)
 *                  Enter ID -> POST /recommend -> cards
 *   4. On card  -> open detail modal
 * ===========================================================
 */

/* --- Config --- */
const API_BASE = "http://localhost:5005";  // Flask server URL (Nuclear Port Migration to 5005)
const BROWSE_PAGE_SIZE = 20;               // Movies shown per "Load more"

/* --- State --- */
let allMovies       = [];   // All movies from /movies
let filteredMovies  = [];   // After search text filter
let browseOffset    = 0;    // Pagination cursor for browse grid
let likedIds        = new Set(); // User's liked movies
let seenIds         = new Set(); // User's seen movies

// --- Poster Batching Manager ---
const PosterBatchManager = {
  queue: [],
  timer: null,
  
  add(title, imgElement, fallbackElement) {
    this.queue.push({ title, imgElement, fallbackElement });
    if (!this.timer) {
      this.timer = setTimeout(() => this.process(), 200);
    }
  },
  
  async process() {
    const currentBatch = this.queue.splice(0, 10);
    this.timer = null;
    if (currentBatch.length === 0) return;
    
    const titles = currentBatch.map(item => item.title);
    try {
      const response = await fetch(`${API_BASE}/api/posters/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ titles })
      });
      const data = await response.json();
      
      currentBatch.forEach(item => {
        const url = data[item.title];
        if (url && url !== "NOT_FOUND") {
          // Use our binary proxy instead of hitting TMDB directly
          item.imgElement.src = `${API_BASE}/api/poster/${encodeURIComponent(item.title)}`;
        } else {
          // Trigger error manually to show fallback
          item.imgElement.onerror();
        }
      });
    } catch (e) {
      console.error("Batch poster error:", e);
      currentBatch.forEach(item => item.imgElement.onerror());
    }
    
    // Continue if more in queue
    if (this.queue.length > 0) {
      this.timer = setTimeout(() => this.process(), 100);
    }
  }
};
let lastRecommendations = [];

/* --- DOM refs --- */
// Shared
const spinner        = document.getElementById("spinner");
const statusText     = document.getElementById("status-text");
const movieGrid      = document.getElementById("movie-grid");
const resultsHeader  = document.getElementById("results-header");
const resultsTitle   = document.getElementById("results-title");
const resultsCount   = document.getElementById("results-count");
const browseGrid     = document.getElementById("browse-grid");
const movieSearch    = document.getElementById("movie-search");
const loadMoreBtn    = document.getElementById("load-more-btn");
const modalTemplate  = document.getElementById("modal-template");
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

/* ==========================================================
   STARTUP - fetch all movies
========================================================== */
async function loadAllMovies() {
  try {
    const res = await fetch(`${API_BASE}/movies`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allMovies      = await res.json();
    filteredMovies = [...allMovies];
    renderBrowseGrid(true);

  } catch (err) {
    console.error("Could not load movies:", err);
    setStatus("⚠️ Could not reach the API. Is Flask initializing or running? (Port 5005)", "error");
  }
}

/* --- Search logic --- */
function applyFilters() {
  const query = movieSearch.value.toLowerCase().trim();

  filteredMovies = allMovies.filter(m => {
    const matchesSearch = !query ||
      m.title.toLowerCase().includes(query) ||
      m.genres.toLowerCase().includes(query);
    return matchesSearch;
  });

  browseOffset = 0;
  renderBrowseGrid(true);
}

/* ==========================================================
   RENDER - Browse Grid
========================================================== */
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

/* --- MODE TOGGLE --- */
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

tabMovie?.addEventListener("click", () => switchMode("movie"));
tabUser?.addEventListener("click", () => switchMode("user"));

/* --- AUTOCOMPLETE (movie title search) --- */
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

/* --- COLD START: Recommend by Movie Title --- */
async function getRecommendationsByMovie(title) {
  if (!title || !title.trim()) {
    setStatus("Please type a movie title first.", "error");
    return;
  }

  setLoading(true);
  setStatus(`Analyzing complex connections... (This takes 20-40s)`);
  movieGrid.innerHTML = "";
  resultsHeader.hidden = true;
  becauseBanner.hidden = true;

  try {
    const mode = document.querySelector('input[name="reco-mode"]:checked')?.value || 'default';
    
    const res = await fetch(`${API_BASE}/recommend-by-movie`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ 
        movie_title: title,
        user_context: {
          mode: mode,
          liked_ids: Array.from(likedIds),
          seen_ids: Array.from(seenIds)
        }
       }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.description || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const { source_movie, recommendations } = data;
    lastRecommendations = recommendations;

    if (!recommendations.length) {
      setStatus("No similar movies found.", "error");
      return;
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

/* --- RECOMMEND - call /recommend endpoint (User ID mode) --- */
async function getRecommendations() {
  const userId = parseInt(userInput.value, 10);
  if (isNaN(userId) || userId < 1) {
    setStatus("⚠️  Please enter a valid User ID (1–610).", "error");
    return;
  }

  setLoading(true);
  setStatus(`🧠 Analyzing top picks for User ${userId}...`);
  movieGrid.innerHTML = "";
  resultsHeader.hidden = true;
  becauseBanner.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/recommend`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ 
        user_id: userId
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.description || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const { recommendations } = data;
    lastRecommendations = recommendations;

    if (!recommendations.length) {
      setStatus("⚠️  No recommendations found for this user.", "error");
      return;
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

/* --- BUILD - Movie Card element --- */
/**
 * createMovieCard(movie, predictedRating, index)
 *
 * @param {Object} movie            — { id, title, genres }
 * @param {number|null} predictedRating — if null, show as browse card
 * @param {number} index            — for animation stagger
 */
function createMovieCard(movie, predictedRating, index) {
  const { id, title, genres, reason, rank } = movie;
  const card = document.createElement("article");
  card.className = "movie-card";
  card.tabIndex  = 0;
  card.setAttribute("aria-label", title);

  // 1. Poster Wrapper
  const posterWrapper = document.createElement("div");
  posterWrapper.className = "card-poster-wrapper";

  // 1a. Generative Fallback
  const fallback = document.createElement("div");
  fallback.className = "generative-poster";
  fallback.id = `poster-fallback-${id}-${index}`;

  const genreList = genres.split("|");
  const primaryGenre = genreList[0];
  const genresMap = {
    "Action": "", "Adventure": "", "Animation": "", "Comedy": "", "Crime": "",
    "Documentary": "", "Drama": "", "Fantasy": "", "Film-Noir": "", "Horror": "",
    "Musical": "", "Mystery": "", "Romance": "", "Sci-Fi": "", "Thriller": "",
    "War": "", "Western": "", "default": ""
  };
  const icon = genresMap[primaryGenre] || genresMap["default"];
  
  const grads = [
    "#5c6e58",
    "#4a5e4b",
    "#8b7d6b",
    "#c4b581"
  ];
  const gradient = grads[Math.abs(hashString(title)) % grads.length];

  fallback.innerHTML = `
    <div class="gp-bg" style="background: ${gradient}"></div>
    <div class="gp-icon">${icon}</div>
    <div class="gp-title">${title}</div>
  `;

  // 1b. Real Image (Batched Discovery)
  const img = document.createElement("img");
  img.className = "poster-img";
  img.alt = title;
  img.loading = "lazy";
  img.style.opacity = "0"; // Fade in on load
  
  img.onload = () => {
    img.style.opacity = "1";
    fallback.style.display = "none";
  };
  img.onerror = () => {
    img.style.display = "none";
    fallback.style.display = "flex";
  };

  // Add to batching manager instead of direct src setting
  PosterBatchManager.add(title, img, fallback);

  posterWrapper.appendChild(fallback);
  posterWrapper.appendChild(img);

  // 2. Card Body
  const body = document.createElement("div");
  body.className = "card-body";

  if (predictedRating !== null && rank) {
    const rankBadge = document.createElement("div");
    rankBadge.className = "card-rank";
    rankBadge.textContent = `#${rank} Pick`;
    body.appendChild(rankBadge);
  }

  const titleEl = document.createElement("h3");
  titleEl.className = "card-title";
  titleEl.textContent = title;
  body.appendChild(titleEl);

  const genreWrap = document.createElement("div");
  genreWrap.className = "card-genres";
  genreList.slice(0, 3).forEach(g => {
    if (g && g !== "(no genres listed)") {
      const badge = document.createElement("span");
      badge.className = "genre-badge";
      badge.textContent = g;
      genreWrap.appendChild(badge);
    }
  });
  body.appendChild(genreWrap);

  if (predictedRating !== null) {
    const ratingWrap = document.createElement("div");
    ratingWrap.className = "card-rating";
    ratingWrap.innerHTML = `
      <span class="stars">${starsForRating(predictedRating)}</span>
      <span class="rating-num">${predictedRating.toFixed(1)} / 5</span>
    `;
    body.appendChild(ratingWrap);
  }

  if (reason && reason !== "INVALID") {
    const reasonEl = document.createElement("p");
    reasonEl.className = "card-reason";
    reasonEl.textContent = reason;
    body.appendChild(reasonEl);
  }

  card.appendChild(posterWrapper);
  card.appendChild(body);

  // 3. Like Button
  const likeBtn = document.createElement("button");
  likeBtn.className = "card-like-btn";
  likeBtn.innerHTML = "❤";
  likeBtn.title = "Mark as liked to personalize results";
  if (likedIds.has(id)) likeBtn.classList.add("liked");
  
  likeBtn.onclick = (e) => {
    e.stopPropagation();
    if (likedIds.has(id)) {
      likedIds.delete(id);
      likeBtn.classList.remove("liked");
    } else {
      likedIds.add(id);
      likeBtn.classList.add("liked");
    }
  };
  card.appendChild(likeBtn);

  const openModal = () => showModal(movie, predictedRating);
  card.addEventListener("click", openModal);
  card.addEventListener("keydown", e => { if (e.key === "Enter") openModal(); });

  return card;
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return hash;
}

/* --- MODAL --- */
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
    rating.hidden = true;
  }

  // 3b. Reason
  const reasonEl = activeModal.querySelector("#modal-reason");
  if (movie.reason && movie.reason !== "INVALID") {
    reasonEl.textContent = movie.reason;
    reasonEl.hidden = false;
  } else {
    reasonEl.hidden = true;
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

/* --- HELPERS --- */
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
  "Action":     "#5c6e58",
  "Adventure":  "#4a5e4b",
  "Animation":  "#8b7d6b",
  "Comedy":     "#c4b581",
  "Crime":      "#3d4d3e",
  "Drama":      "#6e6255",
  "Fantasy":    "#a39373",
  "Horror":     "#2c2f2a",
  "Romance":    "#a07d83",
  "Sci-Fi":     "#5d758f",
  "Thriller":   "#3c4b57",
  "Western":    "#875a46",
  "Musical":    "#a69c6b",
  "Documentary":"#475b6e",
};

function genreColor(genres) {
  const firstGenre = genres.split("|")[0];
  return GENRE_COLORS[firstGenre] || "#5c6e58";
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
  if (recommendBtn) recommendBtn.disabled = on;
  if (!on && spinner.hidden) {
    // no-op, spinner already hidden
  }
}

/* --- EVENT LISTENERS --- */
if (recommendBtn) recommendBtn.addEventListener("click", getRecommendations);
recommendMovieBtn.addEventListener("click", () => {
  getRecommendationsByMovie(movieTitleInput.value);
});

if (userInput) userInput.addEventListener("keydown", e => {
  if (e.key === "Enter") getRecommendations();
});

loadMoreBtn.addEventListener("click", () => renderBrowseGrid(false));
movieSearch.addEventListener("input", debounce(applyFilters, 250));



/* Utility: debounce search input so we don't re-render on every keystroke */
function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

/* --- Bootstrap --- */
document.addEventListener("DOMContentLoaded", () => {
  loadAllMovies();
});
