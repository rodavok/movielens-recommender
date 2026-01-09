# movielens-recommender
Exploring how to build a recommender system using the MovieLens dataset.
=======
# MovieLens 100K Recommender System

Recommender analysis on the MovieLens 100K dataset using ALS and XGBoost.

## Dataset

[MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)

100,000 ratings from 943 users on 1,682 movies. Ratings are 1-5 stars.

Additional data covering the genre and release years for each movie, plus seperate data covering the age and location of the users.

## Approach

Hybrid model combining collaborative filtering and gradient boosting:

1. **ALS (Matrix Factorization)** - Learns latent factors for users and items
2. **XGBoost** - Uses SVD embeddings as features to capture non-linear patterns

### Features Used

- User latent factors (50 dimensions)
- Item latent factors (50 dimensions)
- Element-wise interactions between user/item factors
- ALS prediction as a feature

## Setup

```bash
pip install implicit xgboost pandas numpy scikit-learn
```

Download the dataset from Kaggle and extract to your working directory.

Expected output:
- Model performance metrics (RMSE, MAE)
- Feature importance ranking
- Comparison between pure ALS and hybrid approach
- Comparison between just using user ratings, or that data combined with the genre tags and user info.

## Results

The hybrid ALS + XGBoost model shows the importance of proper train/test splitting to avoid data leakage. Pure ALS typically achieves RMSE ~0.93-0.94, with XGBoost providing marginal improvements by learning non-linear interactions.

Key insight: ALS prediction itself dominates feature importance, suggesting collaborative filtering captures most of the signal in this dataset.

## Notes

- Ensure train/test split happens before SVD training to prevent leakage
- XGBoost predictions are clipped to valid rating range (1-5)
- Feature engineering matters: interactions between latent factors help model performance

## Libraries

- `surprise` - Collaborative filtering algorithms - This package is out of date, and isn't supported by Kaggle anymore.
- `implicit` - Newer library implementing ALS, an alternative to SVD. It's not included in the kaggle kernel, but it is compatible, so it just needs to be installed prior to use.
- `xgboost` - Gradient boosting 
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Evaluation metrics and utilities
