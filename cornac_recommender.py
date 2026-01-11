"""
MovieLens Recommender using Cornac

A collaborative filtering recommender system using the Cornac framework.
Uses MovieLens 100K dataset from Kaggle.
"""

import pandas as pd
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

# Data paths (Kaggle)
DATA_PATH = '/kaggle/input/movielens-100k-dataset/ml-100k'


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


def main():
    # Load ratings
    print("Loading MovieLens 100K ratings...")
    ratings = load_ratings()

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

    # Define collaborative filtering models
    models = [
        MF(k=50, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=42, name="MF"),
        PMF(k=50, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=42, name="PMF"),
        BPR(k=50, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=42, name="BPR"),
    ]

    # Define evaluation metrics
    metrics = [
        MAE(),
        RMSE(),
        Precision(k=10),
        Recall(k=10),
        NDCG(k=10),
        AUC(),
        MAP(),
    ]

    # Run experiment
    print("\nRunning experiment...")
    print("=" * 60)
    cornac.Experiment(
        eval_method=rs,
        models=models,
        metrics=metrics,
        user_based=True,
    ).run()


if __name__ == "__main__":
    main()
