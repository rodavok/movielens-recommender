"""
MovieLens Recommender using Cornac

A collaborative filtering recommender system using the Cornac framework.
Uses MovieLens 100K dataset from Kaggle.
Compares models with and without genre information.
"""

import pandas as pd
import numpy as np
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR, ItemKNN, MMMF
from cornac.data import FeatureModality
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

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


def run_ratings_only_experiment(ratings):
    """Run experiment using only user-movie ratings (no genre info)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Ratings Only (Collaborative Filtering)")
    print("=" * 70)

    # Split data
    print("\nSplitting data (80/20)...")
    rs = RatioSplit(
        data=ratings,
        test_size=0.2,
        rating_threshold=4.0,
        exclude_unknowns=True,
        seed=42,
        verbose=True
    )

    # Define collaborative filtering models (no side information)
    models = [
        MF(k=50, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=42, name="MF"),
        PMF(k=50, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=42, name="PMF"),
        BPR(k=50, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=42, name="BPR"),
    ]

    # Run experiment
    print("\nRunning experiment (ratings only)...")
    exp = cornac.Experiment(
        eval_method=rs,
        models=models,
        metrics=get_metrics(),
        user_based=True,
    )
    exp.run()
    return exp


def run_genre_experiment(ratings, item_ids, genre_features):
    """Run experiment using ratings + genre information."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Ratings + Genre Information")
    print("=" * 70)

    # Create item feature modality for genre information
    item_feature = FeatureModality(
        features=genre_features,
        ids=item_ids,
        normalize=True
    )

    # Split data with item features
    print("\nSplitting data (80/20) with genre features...")
    rs = RatioSplit(
        data=ratings,
        test_size=0.2,
        rating_threshold=4.0,
        exclude_unknowns=True,
        seed=42,
        verbose=True,
        item_feature=item_feature
    )

    # Models that can leverage item features
    # ItemKNN uses features for similarity, MMMF jointly factorizes ratings and features
    models = [
        MF(k=50, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=42, name="MF"),
        PMF(k=50, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=42, name="PMF"),
        BPR(k=50, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=42, name="BPR"),
        ItemKNN(k=50, similarity='cosine', name="ItemKNN-Genre"),
        MMMF(k=50, max_iter=100, learning_rate=0.001, lambda_reg=0.01, seed=42, name="MMMF-Genre"),
    ]

    # Run experiment
    print("\nRunning experiment (with genre features)...")
    exp = cornac.Experiment(
        eval_method=rs,
        models=models,
        metrics=get_metrics(),
        user_based=True,
    )
    exp.run()
    return exp


def print_comparison(exp1, exp2):
    """Print side-by-side comparison of two experiments."""
    print("\n" + "=" * 70)
    print("COMPARISON: Impact of Genre Information")
    print("=" * 70)

    # Get common models (MF, PMF, BPR)
    common_models = ['MF', 'PMF', 'BPR']
    metrics_to_compare = ['NDCG@10', 'Precision@10', 'Recall@10', 'MAP', 'AUC']

    print("\nCommon models (MF, PMF, BPR) - with vs without genre features:")
    print("-" * 70)

    for metric_name in metrics_to_compare:
        print(f"\n{metric_name}:")
        print(f"  {'Model':<10} {'Ratings Only':>15} {'With Genre':>15} {'Difference':>15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")

        for model_name in common_models:
            # Find results in both experiments
            val1 = None
            val2 = None

            for result in exp1.result:
                if result.model_name == model_name:
                    for metric_key, value in result.metric_avg_results.items():
                        key_name = metric_key.name if hasattr(metric_key, 'name') else str(metric_key)
                        if key_name == metric_name:
                            val1 = value
                            break

            for result in exp2.result:
                if result.model_name == model_name:
                    for metric_key, value in result.metric_avg_results.items():
                        key_name = metric_key.name if hasattr(metric_key, 'name') else str(metric_key)
                        if key_name == metric_name:
                            val2 = value
                            break

            if val1 is not None and val2 is not None:
                diff = val2 - val1
                diff_pct = (diff / val1 * 100) if val1 != 0 else 0
                sign = '+' if diff >= 0 else ''
                print(f"  {model_name:<10} {val1:>15.4f} {val2:>15.4f} {sign}{diff:>14.4f} ({sign}{diff_pct:.1f}%)")

    # Show genre-specific models
    print("\n" + "-" * 70)
    print("Genre-specific models (only available with genre features):")
    genre_models = ['ItemKNN-Genre', 'MMMF-Genre']

    for result in exp2.result:
        if result.model_name in genre_models:
            print(f"\n{result.model_name}:")
            for metric_key, value in result.metric_avg_results.items():
                key_name = metric_key.name if hasattr(metric_key, 'name') else str(metric_key)
                if key_name in metrics_to_compare:
                    print(f"  {key_name}: {value:.4f}")


def main():
    # Load ratings
    print("Loading MovieLens 100K ratings...")
    ratings = load_ratings()

    # Load genre information
    item_ids, genre_features = load_item_genres()

    # Run both experiments
    exp1 = run_ratings_only_experiment(ratings)
    exp2 = run_genre_experiment(ratings, item_ids, genre_features)

    # Print comparison
    print_comparison(exp1, exp2)


if __name__ == "__main__":
    main()
