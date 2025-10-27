## 🎬 Movie Analysis

### 🎯 Goal
- Identify clear patterns between outlier and non-outlier movies based on genre and release month.
- Understand the relationship between cast and crew ratings, budget, and movie ratings/revenue.

### 🧩 Approach

**IMDb Data Preparation**
- Retrieved IMDb datasets using SQL queries and merged key tables to obtain full movie-level information.
- Calculated individual creative role ratings (composer, director, writer, cinematographer, editor) using a weighted average of IMDb ratings across all their movies.
- Filtered data to include only feature films (2018–2025) with ≥ 5,000 votes to ensure audience significance.

**TMDb Data Integration**
- Extracted complementary data (e.g., budget, revenue) from the TMDb API.
- Matched TMDb and IMDb datasets via title and IMDb ID for consistency.

**Data Cleaning and Consolidation**
- Combined IMDb and TMDb data into a unified dataset.
- Removed invalid or missing entries (e.g., negative budgets/revenues).
- Standardized column names and merged multiple genre columns into one.
- Final dataset included:
['title', 'imdb_id', 'release_date', 'runtime', 'budget', 'revenue',
 'tmdb_rating', 'vote_count', 'runtimeMinutes', 'imdb_rating',
 'avg_cast_rating', 'director_rating', 'writer_rating',
 'composer_rating', 'cinematographer_rating', 'editor_rating', 'genre']

### 🔍 Data Analysis

**Part 1 — Unsupervised Learning**
- Converted release_date to datetime and extracted release month.
- Selected key columns: 'title', 'release_month', 'runtimeMinutes', 'genre', 'budget', 'revenue'.
- Applied One-Hot Encoding for months and genres.
- Scaled numerical features using StandardScaler (retaining the impact of budget/revenue for blockbuster detection).
- Applied KMeans clustering (k=3) to segment movies.
- Performed Association Rule Mining on clusters:
- Cluster 2 → highest average revenue (blockbusters)
- Cluster 0 & 1 → non-blockbusters
- Filtered rules where antecedents = genre and consequents = release month.

**Part 2 — Supervised Learning**
- Dropped missing values in cast/crew ratings.
- Features:
['runtimeMinutes', 'vote_count', 'director_rating', 'writer_rating', 'composer_rating', 'cinematographer_rating', 'editor_rating', 'tmdb_rating', 'imdb_rating', 'revenue', 'budget']
- Target Variables:
['tmdb_rating', 'imdb_rating']
- Scaled features and trained a MultiOutput Linear Regression model.
- Identified top-performing genres by weighting their total appearances and mean ratings/revenues.

### 📈 Results

**Unsupervised Learning**
- High-budget blockbusters cluster around June, with some peaks in December and February.
- Low-budget films show no consistent release-month pattern.

**Supervised Learning**
- Ratings (IMDb/TMDb) strongly correlate with cast and crew quality, especially writers and directors.
- Budget and revenue are weak predictors of ratings, but vote count strongly predicts commercial success.

Genres:
- Action, Adventure, Sci-Fi → dominate revenue
- Drama, Comedy, Romance, Sport → dominate ratings

### 🧠 Insights
- Release strategy matters: Studios time big-budget films (especially action/adventure) for high-grossing months like June and December.
- Popularity ≠ Quality: While budget and exposure drive revenue, cast/crew skill predicts critical reception.
- Audience engagement (vote count) acts as a bridge between financial and critical success.
- Potential research expansion: integrate social media sentiment and trailer analytics to improve box office predictions.

### 🛠️ Tech Stack
Python | Pandas | DuckDB | Matplotlib | Scikit-learn

### 🧭 Next Steps
Extend dataset with YouTube clip analytics (views, likes, sentiment).

Build emotion-based soundtrack prediction using Librosa + Transformer architectures.
