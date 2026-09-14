# Music Streaming Predictor

Predicts a song's Spotify stream count from its audio features, using the
Kaggle Spotify and YouTube dataset. Built for Track H (AI Foundations),
Gate H2 — Python for ML & First Pipeline.

## What this does

Loads raw Spotify/YouTube data → cleans it → splits train/test → builds a
scikit-learn preprocessing + regression pipeline → evaluates on held-out data.

- **Target**: `Stream` (log-transformed, since raw stream counts are heavily
  right-skewed)
- **Features**: audio features (Danceability, Energy, Loudness, Speechiness,
  Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration_ms),
  plus `Key` and `Album_type` as categoricals
- **Excluded as leakage**: `Views`, `Likes`, `Comments` — these are YouTube
  popularity signals correlated with a song already being successful, not
  properties known independent of that
- **Model**: `RandomForestRegressor` inside a `ColumnTransformer` +
  `Pipeline`, with `StandardScaler` on numeric features and `OneHotEncoder`
  on categoricals

## Setup

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install pandas numpy scikit-learn jupyter
```

Download `Spotify_Youtube.csv` from Kaggle and place it in the project root
(it's gitignored — not committed).

## Running

```bash
jupyter notebook spotify_youtube_ml_pipeline.ipynb
```

Restart kernel → Run All should execute top to bottom with no errors and no
manual steps.

## Results

Baseline `RandomForestRegressor` on audio features alone:

| Metric | Value |
|---|---|
| R² | 0.24 |
| RMSE (log-space) | 1.46 |
| RMSE (back-transformed streams) | ~235.5M |

Audio features alone explain roughly a quarter of stream-count variance —
expected for a first baseline, since popularity is driven by far more than
danceability/energy/etc. (marketing, release timing, artist fanbase, virality).

## Known cleaning decisions

- `Key == -1` (no key detected) treated as its own category, not a numeric
  value
- Rows with nulls in any feature column or the target dropped
- `Unnamed: 0` (raw CSV index) dropped
- High-cardinality identifier columns (`Artist`, `Track`, `Album`, `Channel`,
  URLs, `Title`, `Description`) dropped as non-predictive for a first pipeline
- Duplicate `Artist`+`Track` rows: [fill in your decision once made — kept
  because X, or dropped because Y]

## Branch structure

Work for this gate is organized as: `main` ← `feature/music-streaming-predictor`
(integration) ← `music/setup`, `music/exploration`, `music/cleaning`,
`music/split`, `music/pipeline-train`, `music/evaluation`, `music/writeup`
(phase branches, merged via `--no-ff`).