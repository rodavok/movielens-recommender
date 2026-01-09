# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a recommender system project using the MovieLens 100K dataset. The implementation combines collaborative filtering (ALS) with gradient boosting (XGBoost) to predict movie ratings, and compares performance with and without movie metadata features.

https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset

## Key Architecture

### Hybrid Model Pipeline

The recommender system compares 4 models:

1. **Model 1: Pure ALS** - Baseline collaborative filtering using Alternating Least Squares matrix factorization
2. **Model 2: XGBoost + ALS Features** - Uses 151 ALS-derived features (50 user + 50 item + 50 interactions + 1 ALS prediction)
3. **Model 3: Pure ALS** - Same as Model 1 (included for comparison symmetry)
4. **Model 4: XGBoost + ALS + Metadata** - Uses 171 features (151 ALS + 19 genre features + 1 release year)

### Collaborative Filtering: ALS (Alternating Least Squares)

**ALS** is a matrix factorization technique that learns 50-dimensional latent factors for users and items through alternating optimization:
- Fixes user factors, optimizes item factors
- Fixes item factors, optimizes user factors
- Repeats for 20 iterations

**Why ALS instead of SVD?** The `implicit` library (ALS) is actively maintained with NumPy 2.0 support, while `scikit-surprise` (SVD) has compatibility issues with modern dependencies.

### Feature Engineering

**Model 2 (ALS-only)** - 151 features:
- User latent factors (50 dimensions)
- Item latent factors (50 dimensions)
- Element-wise interactions between user/item factors (50 dimensions)
- ALS prediction score via dot product (1 feature)

**Model 4 (ALS + Metadata)** - 171 features:
- All 151 ALS features above
- 19 genre features (one-hot encoded: action, comedy, drama, sci-fi, etc.)
- 1 normalized release year feature (min-max scaled)

### Data Leakage Prevention

**CRITICAL**: The train/test split happens BEFORE ALS training. This ensures the ALS model only sees training data and prevents leakage. User/item ID mappings are created from the training set and used for feature extraction on both train and test sets.

## Development Commands

### Setup

Install dependencies:
```bash
pip install implicit xgboost pandas numpy scikit-learn scipy
```

**Note**: The project uses `implicit` library for ALS instead of `scikit-surprise` for SVD, as it's actively maintained and compatible with NumPy 2.0+.

### Running the Analysis

Execute the Python pipeline:
```bash
python movielens-reccomender-pipeline.py
```

Or execute the Jupyter notebook:
```bash
jupyter notebook movielens-reccomender.ipynb
```

## Dataset Structure

The project uses the MovieLens 100K dataset (100,000 ratings from 943 users on 1,682 movies):

- **u.data** - User ratings (user_id, item_id, rating, timestamp)
- **u.item** - Movie metadata (item_id, name, release_date, genres as one-hot)
- **u.user** - User information (user_id, age, gender, occupation, zip_code)

Expected dataset location: `/kaggle/input/movielens-100k-dataset/ml-100k/`

## Code Structure

The pipeline (`movielens-reccomender-pipeline.py`) follows this sequential flow:

1. **Load data**: user_ratings from u.data, movie_metadata from u.item
2. **Split data**: Train/test (80/20) to prevent leakage
3. **Train ALS**:
   - Create user/item ID mappings (external ID → internal index)
   - Build sparse CSR matrix (item × user format)
   - Train ALS model on training set only
4. **Extract ALS features**: Using `get_als_features()` helper (151 dimensions)
5. **Extract enhanced features**: Using `get_enhanced_features()` helper (171 dimensions with metadata)
6. **Train XGBoost models**: Two variants (ALS-only and ALS+Metadata)
7. **Evaluate all 4 models**: Pure ALS (×2), XGBoost+ALS, XGBoost+ALS+Metadata
8. **Analyze feature importance**: Identify top features and metadata contribution
9. **Display comparison table**: Side-by-side RMSE/MAE with improvement percentages
10. **Show sample predictions**: First 5 test cases across all models

### Key Helper Functions

- `extract_release_year()` - Parses date strings, returns year (defaults to 1995)
- `create_movie_features_lookup()` - Builds dict of 20 metadata features per movie
- `get_als_features()` - Extracts 151 ALS-derived features for a user-item pair
- `get_enhanced_features()` - Combines ALS features with movie metadata (171 total)
- `evaluate_model()` - Calculates RMSE and MAE with prediction clipping

## Important Implementation Details

- **Rating Range**: Predictions are clipped to valid range [1, 5]
- **ALS Parameters**: 50 factors, 20 iterations, random_state=42
- **XGBoost Parameters**: 100 estimators, max_depth=6, learning_rate=0.1
- **Error Handling**: Feature extraction functions return None for unknown user/item pairs
- **Sparse Matrix Format**: Item×user (implicit library convention), not user×item
- **ID Mapping**: External IDs mapped to internal indices via dictionaries for O(1) lookup
- **Performance**:
  - Model 2 (XGBoost+ALS) typically shows ~24% RMSE improvement over pure ALS
  - Model 4 (XGBoost+ALS+Metadata) expected to show 2-3% additional improvement

## Model Interpretation

The feature importance analysis consistently shows that `als_prediction` dominates (typically 35-40% importance), indicating that collaborative filtering captures most of the signal. The interaction features provide marginal improvements by learning non-linear patterns between user and item embeddings.

**Metadata contribution**: Genre and release year features typically contribute 5-10% of total importance, with drama, comedy, and action being the most predictive genres. This suggests content-based features provide modest but measurable improvements over pure collaborative filtering.
