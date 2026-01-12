"""
MovieLens Recommender using Cornac + XGBoost

A hybrid recommender system comparing SVD alone vs SVD + XGBoost.
Uses MovieLens 100K dataset from Kaggle.
"""

import pandas as pd
import numpy as np
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import SVD
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# Data paths (Kaggle)
DATA_PATH = '/kaggle/input/movielens-100k-dataset/ml-100k'

# Genre names in MovieLens 100K (19 genres)
GENRE_NAMES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def load_ratings():
    """Load user ratings from u.data file."""
    user_ratings = pd.read_csv(
        f'{DATA_PATH}/u.data',
        sep='\t',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    print(f"Loaded {len(user_ratings)} ratings")
    print(f"  - {user_ratings['user_id'].nunique()} users")
    print(f"  - {user_ratings['item_id'].nunique()} items")

    # Convert to Cornac format: list of (user_id, item_id, rating) tuples
    # Cornac expects string IDs
    ratings = [
        (str(row['user_id']), str(row['item_id']), float(row['rating']))
        for _, row in user_ratings.iterrows()
    ]
    return ratings


def load_item_genres():
    """Load item genre information from u.item file.

    Returns:
        tuple: (item_ids, genre_features) where:
            - item_ids: list of item IDs as strings
            - genre_features: numpy array of shape (n_items, 19) with binary genre indicators
    """
    # u.item format: movie_id|title|release_date|video_release_date|url|genre1|...|genre19
    # Columns 5-23 (0-indexed) are the 19 genre binary indicators
    columns = ['item_id', 'title', 'release_date', 'video_release_date', 'url'] + GENRE_NAMES

    items = pd.read_csv(
        f'{DATA_PATH}/u.item',
        sep='|',
        header=None,
        names=columns,
        encoding='latin-1'  # MovieLens uses latin-1 encoding
    )

    print(f"\nLoaded genre data for {len(items)} items")

    # Extract item IDs and genre features
    item_ids = [str(iid) for iid in items['item_id'].values]
    genre_features = items[GENRE_NAMES].values.astype(np.float32)

    # Print genre distribution
    genre_counts = genre_features.sum(axis=0)
    print("Genre distribution:")
    for name, count in sorted(zip(GENRE_NAMES, genre_counts), key=lambda x: -x[1])[:5]:
        print(f"  - {name}: {int(count)} movies")

    return item_ids, genre_features


def get_metrics():
    """Get evaluation metrics used for all experiments."""
    return [
        MAE(),
        RMSE(),
        Precision(k=10),
        Recall(k=10),
        NDCG(k=10),
        AUC(),
        MAP(),
    ]


def get_svd_factors(svd_model):
    """Get user and item factor matrices from SVD model."""
    # Cornac SVD uses u_factors and i_factors
    return svd_model.u_factors, svd_model.i_factors


def build_feature_matrix_from_cornac(df, svd_model, train_set, genre_dict):
    """Build feature matrix from SVD embeddings and genre features."""
    user_factors_matrix, item_factors_matrix = get_svd_factors(svd_model)

    features = []
    for _, row in df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']

        # Get SVD embeddings using Cornac's internal indices
        try:
            user_idx = train_set.uid_map[user_id]
            item_idx = train_set.iid_map[item_id]
            user_factors = user_factors_matrix[user_idx]
            item_factors = item_factors_matrix[item_idx]
        except (KeyError, IndexError):
            # Unknown user or item - use zeros
            user_factors = np.zeros(svd_model.k)
            item_factors = np.zeros(svd_model.k)

        # Get genre features
        genre_feats = genre_dict.get(item_id, np.zeros(19))

        # Combine: user factors + item factors + genres
        feat = np.concatenate([user_factors, item_factors, genre_feats])
        features.append(feat)

    return np.array(features)


def run_svd_experiment(ratings):
    """Run SVD-only experiment using Cornac.

    Uses 60/20/20 split: train SVD on 60%, validate XGBoost on 20%, test both on 20%.
    """
    print("\n" + "=" * 70)
    print("MODEL 1: SVD Only")
    print("=" * 70)

    # Create evaluation method with 60/20/20 split (val_size is relative to remaining after test)
    # test_size=0.2 takes 20% for test, val_size=0.25 takes 25% of remaining 80% = 20% for val
    rs = RatioSplit(
        data=ratings,
        test_size=0.2,
        val_size=0.25,  # 25% of remaining 80% = 20% of total for validation
        rating_threshold=4.0,
        exclude_unknowns=True,
        seed=42,
        verbose=True
    )

    # Train SVD
    svd_model = SVD(k=50, max_iter=100, learning_rate=0.01, lambda_reg=0.02, seed=42, name="SVD")

    print("\nTraining SVD...")
    exp = cornac.Experiment(
        eval_method=rs,
        models=[svd_model],
        metrics=get_metrics(),
        user_based=True,
    )
    exp.run()

    return exp, svd_model, rs


def run_svd_xgboost_experiment(ratio_split, svd_model, genre_dict):
    """Run SVD + XGBoost hybrid experiment.

    XGBoost is trained on validation set (data SVD hasn't seen during training)
    and evaluated on test set (same as SVD evaluation).
    """
    print("\n" + "=" * 70)
    print("MODEL 2: SVD + XGBoost (Hybrid)")
    print("=" * 70)

    train_set = ratio_split.train_set
    val_set = ratio_split.val_set
    test_set = ratio_split.test_set

    # Build feature matrices from Cornac's val/test sets
    print("\nBuilding feature matrices from SVD embeddings + genres...")
    print("  (XGBoost trains on validation set - data SVD hasn't seen)")

    # Create reverse mappings (index -> original ID)
    idx_to_uid = {idx: uid for uid, idx in train_set.uid_map.items()}
    idx_to_iid = {idx: iid for iid, idx in train_set.iid_map.items()}

    # Extract validation data for XGBoost training
    val_user_indices, val_item_indices, val_ratings = val_set.uir_tuple
    val_uids = [idx_to_uid.get(i, None) for i in val_user_indices]
    val_iids = [idx_to_iid.get(i, None) for i in val_item_indices]

    val_df = pd.DataFrame({
        'user_id': val_uids,
        'item_id': val_iids,
        'rating': val_ratings
    })

    # Extract test data for evaluation
    test_user_indices, test_item_indices, test_ratings = test_set.uir_tuple
    test_uids = [idx_to_uid.get(i, None) for i in test_user_indices]
    test_iids = [idx_to_iid.get(i, None) for i in test_item_indices]

    test_df = pd.DataFrame({
        'user_id': test_uids,
        'item_id': test_iids,
        'rating': test_ratings
    })

    # XGBoost trains on validation set, tests on test set
    X_train = build_feature_matrix_from_cornac(val_df, svd_model, train_set, genre_dict)
    y_train = val_df['rating'].values
    X_test = build_feature_matrix_from_cornac(test_df, svd_model, train_set, genre_dict)
    y_test = test_df['rating'].values

    print(f"  XGBoost train (validation set): {X_train.shape}")
    print(f"  XGBoost test: {X_test.shape}")

    # Hyperparameter tuning for XGBoost
    print("\nTuning XGBoost hyperparameters...")

    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
    }

    base_model = XGBRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)

    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.4f}")

    xgb_model = search.best_estimator_

    # Predict
    y_pred = xgb_model.predict(X_test)
    y_pred = np.clip(y_pred, 1, 5)  # Clip to valid rating range

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nSVD + XGBoost Results:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    return {'MAE': mae, 'RMSE': rmse, 'model': xgb_model}


def print_comparison(svd_exp, xgb_results):
    """Print side-by-side comparison of SVD vs SVD+XGBoost."""
    print("\n" + "=" * 70)
    print("COMPARISON: SVD vs SVD + XGBoost")
    print("=" * 70)

    # Extract SVD metrics
    svd_result = svd_exp.result[0]
    svd_metrics = {}
    for metric_key, value in svd_result.metric_avg_results.items():
        key_name = metric_key.name if hasattr(metric_key, 'name') else str(metric_key)
        svd_metrics[key_name] = value

    print(f"\n{'Metric':<15} {'SVD':>12} {'SVD+XGBoost':>12} {'Difference':>15}")
    print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*15}")

    # Compare rating prediction metrics
    for metric in ['MAE', 'RMSE']:
        svd_val = svd_metrics.get(metric, 0)
        xgb_val = xgb_results.get(metric, 0)
        diff = xgb_val - svd_val
        diff_pct = (diff / svd_val * 100) if svd_val != 0 else 0
        sign = '+' if diff >= 0 else ''
        # For MAE/RMSE, lower is better
        better = "worse" if diff > 0 else "better"
        print(f"{metric:<15} {svd_val:>12.4f} {xgb_val:>12.4f} {sign}{diff:>14.4f} ({better})")

    print("\n(Note: For MAE/RMSE, lower values are better)")
    print("(SVD+XGBoost uses SVD embeddings + genre features as input to XGBoost)")


def main():
    # Load ratings
    print("Loading MovieLens 100K ratings...")
    ratings = load_ratings()

    # Load genre information
    item_ids, genre_features = load_item_genres()

    # Create genre dictionary for quick lookup
    genre_dict = {item_id: genre_features[i] for i, item_id in enumerate(item_ids)}

    # Run SVD experiment (handles its own train/test split)
    svd_exp, svd_model, ratio_split = run_svd_experiment(ratings)

    # Run SVD + XGBoost experiment (uses same split from ratio_split)
    xgb_results = run_svd_xgboost_experiment(ratio_split, svd_model, genre_dict)

    # Print comparison
    print_comparison(svd_exp, xgb_results)


if __name__ == "__main__":
    main()
