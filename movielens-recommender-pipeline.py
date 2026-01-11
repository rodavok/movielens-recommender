import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Helper functions
def extract_release_year(release_date_str):
    """Extract year from 'DD-Mon-YYYY' format, default to 1995 if missing"""
    if pd.isna(release_date_str) or release_date_str == '':
        return 1995
    try:
        date_obj = pd.to_datetime(release_date_str, format='%d-%b-%Y')
        return date_obj.year
    except:
        return 1995

def create_movie_features_lookup(movie_metadata):
    """Returns dict {item_id: np.array([19 genres + 1 normalized_year])}"""
    # Extract and normalize release year
    movie_metadata['release_year'] = movie_metadata['release_date'].apply(extract_release_year)
    min_year = movie_metadata['release_year'].min()
    max_year = movie_metadata['release_year'].max()
    movie_metadata['year_normalized'] = (movie_metadata['release_year'] - min_year) / (max_year - min_year)

    # Genre columns (19 one-hot encoded)
    genre_cols = ['unknown', 'action', 'adventure', 'animation', 'children',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                  'film_noir', 'horror', 'musical', 'mystery', 'romance',
                  'sci_fi', 'thriller', 'war', 'western']

    # Build dictionary
    movie_dict = {}
    for _, row in movie_metadata.iterrows():
        item_id = row['item_id']
        genre_features = row[genre_cols].values.astype(float)
        year_feature = np.array([row['year_normalized']])
        movie_dict[item_id] = np.concatenate([genre_features, year_feature])

    return movie_dict

def get_als_features(user_id, item_id, als_model, user_id_map, item_id_map):
    """Extract ALS latent factors as features"""
    try:
        # Map external IDs to internal indices
        user_idx = user_id_map[user_id]
        item_idx = item_id_map[item_id]

        # Get user and item factors
        user_factors = als_model.user_factors[user_idx]
        item_factors = als_model.item_factors[item_idx]

        # Predict rating via dot product
        als_prediction = np.dot(user_factors, item_factors)

        # Combine features
        features = np.concatenate([
            user_factors,
            item_factors,
            user_factors * item_factors,  # Element-wise interaction
            [als_prediction]  # ALS prediction
        ])
        return features
    except:
        return None

def get_enhanced_features(user_id, item_id, als_model, user_id_map, item_id_map, movie_features_dict):
    """Extract 171 features: 151 ALS + 19 genres + 1 year"""
    # Get base ALS features (151 dims)
    als_features = get_als_features(user_id, item_id, als_model, user_id_map, item_id_map)
    if als_features is None:
        return None

    # Get movie metadata (20 dims)
    movie_features = movie_features_dict.get(item_id, np.zeros(20))

    # Concatenate
    return np.concatenate([als_features, movie_features])

def evaluate_model(y_true, y_pred, model_name):
    """Calculate RMSE and MAE"""
    y_pred_clipped = np.clip(y_pred, 1, 5)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_clipped))
    mae = mean_absolute_error(y_true, y_pred_clipped)
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae}

# 1. Load data into the pipeline

# Load user ratings
user_ratings = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.data',
                           sep='\t',
                           header=None,
                           names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load movie metadata
movie_metadata = pd.read_csv('/kaggle/input/movielens-100k-dataset/ml-100k/u.item',
                             sep='|', header=None,
                             names=['item_id', 'item_name', 'release_date',
                                    'video_release_date', 'imdb_link',
                                    'unknown', 'action', 'adventure', 'animation', 'children',
                                    'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                                    'film_noir', 'horror', 'musical', 'mystery', 'romance',
                                    'sci_fi', 'thriller', 'war', 'western'],
                             dtype={'video_release_date': 'str'},
                             encoding='latin-1')

df = user_ratings

# 2. Split data into training and test sets in order to avoid data leakage
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Train ALS only on training data
# Create ID mappings (external ID -> internal index)
unique_users = sorted(train_df['user_id'].unique())
unique_items = sorted(train_df['item_id'].unique())
user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

# Map training data to internal indices
train_df['user_idx'] = train_df['user_id'].map(user_id_map)
train_df['item_idx'] = train_df['item_id'].map(item_id_map)

# Create sparse matrix (item x user format for implicit)
# implicit expects confidence values, so we use ratings as-is
user_item_matrix = csr_matrix(
    (train_df['rating'].values, (train_df['item_idx'].values, train_df['user_idx'].values)),
    shape=(len(unique_items), len(unique_users))
)

# Train ALS model
als = AlternatingLeastSquares(factors=50, iterations=20, random_state=42)
als.fit(user_item_matrix)

print(f"Trained ALS model with {len(unique_users)} users and {len(unique_items)} items")

# Calibrate ALS predictions to 1-5 scale using training data
# Collect raw ALS predictions on training set
train_als_raw_predictions = []
train_actual_ratings = []

for _, row in train_df.iterrows():
    try:
        user_idx = user_id_map[row['user_id']]
        item_idx = item_id_map[row['item_id']]
        raw_pred = np.dot(als.user_factors[user_idx], als.item_factors[item_idx])
        train_als_raw_predictions.append(raw_pred)
        train_actual_ratings.append(row['rating'])
    except:
        pass

train_als_raw_predictions = np.array(train_als_raw_predictions)
train_actual_ratings = np.array(train_actual_ratings)

# Fit linear calibration: rating = a * raw_pred + b
# Using least squares: minimize sum((a*x + b - y)^2)
X_calib = np.column_stack([train_als_raw_predictions, np.ones(len(train_als_raw_predictions))])
calib_params = np.linalg.lstsq(X_calib, train_actual_ratings, rcond=None)[0]
als_scale, als_bias = calib_params

print(f"ALS calibration parameters: scale={als_scale:.4f}, bias={als_bias:.4f}")

# 4. Create movie features lookup
movie_features_dict = create_movie_features_lookup(movie_metadata)
print(f"Created movie features for {len(movie_features_dict)} movies")

# 5. Create ALS-only features for training set (Model 2)
X_train_als = []
y_train_als = []

for _, row in train_df.iterrows():
    features = get_als_features(row['user_id'], row['item_id'], als, user_id_map, item_id_map)
    if features is not None:
        X_train_als.append(features)
        y_train_als.append(row['rating'])

X_train_als = np.array(X_train_als)
y_train_als = np.array(y_train_als)

# 6. Create ALS-only features for test set (Model 2) and track valid indices
X_test_als = []
y_test_als = []
valid_test_indices = []  # Track which test_df rows have valid features

for idx, row in test_df.iterrows():
    features = get_als_features(row['user_id'], row['item_id'], als, user_id_map, item_id_map)
    if features is not None:
        X_test_als.append(features)
        y_test_als.append(row['rating'])
        valid_test_indices.append(idx)  # Store the original index

X_test_als = np.array(X_test_als)
y_test_als = np.array(y_test_als)

print(f"ALS-only features - Training samples: {len(X_train_als)}, Test samples: {len(X_test_als)}")

# 7. Create enhanced features for training set (Model 4)
X_train_enhanced = []
y_train_enhanced = []

for _, row in train_df.iterrows():
    features = get_enhanced_features(row['user_id'], row['item_id'], als, user_id_map, item_id_map, movie_features_dict)
    if features is not None:
        X_train_enhanced.append(features)
        y_train_enhanced.append(row['rating'])

X_train_enhanced = np.array(X_train_enhanced)
y_train_enhanced = np.array(y_train_enhanced)

# 8. Create enhanced features for test set (Model 4)
X_test_enhanced = []
y_test_enhanced = []

for _, row in test_df.iterrows():
    features = get_enhanced_features(row['user_id'], row['item_id'], als, user_id_map, item_id_map, movie_features_dict)
    if features is not None:
        X_test_enhanced.append(features)
        y_test_enhanced.append(row['rating'])

X_test_enhanced = np.array(X_test_enhanced)
y_test_enhanced = np.array(y_test_enhanced)

print(f"Enhanced features - Training samples: {len(X_train_enhanced)}, Test samples: {len(X_test_enhanced)}")

# 9. Train XGBoost with ALS features (Model 2)
xgb_als = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
xgb_als.fit(X_train_als, y_train_als)

# 10. Train XGBoost with enhanced features (Model 4)
xgb_enhanced = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
xgb_enhanced.fit(X_train_enhanced, y_train_enhanced)

# 11. Evaluate all 4 models
results = []

# Model 1: Pure ALS - only predict for valid test indices (with calibration)
y_pred_als = []
for idx in valid_test_indices:
    row = test_df.loc[idx]
    try:
        user_idx = user_id_map[row['user_id']]
        item_idx = item_id_map[row['item_id']]
        raw_prediction = np.dot(als.user_factors[user_idx], als.item_factors[item_idx])
        # Apply calibration
        calibrated_prediction = als_scale * raw_prediction + als_bias
        y_pred_als.append(calibrated_prediction)
    except:
        # If user/item not in training set, use global mean (shouldn't happen for valid indices)
        y_pred_als.append(train_df['rating'].mean())

y_pred_als = np.array(y_pred_als)
results.append(evaluate_model(y_test_als, y_pred_als, 'Model 1: Pure ALS'))

# Model 2: XGBoost + ALS Features
y_pred_xgb_als = xgb_als.predict(X_test_als)
results.append(evaluate_model(y_test_als, y_pred_xgb_als, 'Model 2: XGBoost + ALS Features'))

# Model 3: Pure ALS (same as Model 1)
results.append(evaluate_model(y_test_als, y_pred_als, 'Model 3: Pure ALS'))

# Model 4: XGBoost + ALS + Metadata
y_pred_xgb_enhanced = xgb_enhanced.predict(X_test_enhanced)
results.append(evaluate_model(y_test_enhanced, y_pred_xgb_enhanced,
                              'Model 4: XGBoost + ALS + Metadata'))

# 12. Display comparison table
results_df = pd.DataFrame(results)

# Add improvement columns
baseline_rmse = results_df.iloc[0]['RMSE']
baseline_mae = results_df.iloc[0]['MAE']
results_df['RMSE_Improvement'] = results_df['RMSE'].apply(
    lambda x: f"{((baseline_rmse - x) / baseline_rmse * 100):.2f}%"
)
results_df['MAE_Improvement'] = results_df['MAE'].apply(
    lambda x: f"{((baseline_mae - x) / baseline_mae * 100):.2f}%"
)

# Format numeric columns
results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.4f}")
results_df['MAE'] = results_df['MAE'].apply(lambda x: f"{x:.4f}")

print("\n" + "="*100)
print("MODEL PERFORMANCE COMPARISON")
print("="*100)
print(results_df.to_string(index=False))
print("="*100)

# 13. Feature importance for enhanced model (Model 4)
feature_names_enhanced = (
    [f'user_factor_{i}' for i in range(50)] +
    [f'item_factor_{i}' for i in range(50)] +
    [f'interaction_{i}' for i in range(50)] +
    ['als_prediction'] +
    ['unknown', 'action', 'adventure', 'animation', 'children',
     'comedy', 'crime', 'documentary', 'drama', 'fantasy',
     'film_noir', 'horror', 'musical', 'mystery', 'romance',
     'sci_fi', 'thriller', 'war', 'western'] +
    ['release_year_normalized']
)

importance_df = pd.DataFrame({
    'feature': feature_names_enhanced,
    'importance': xgb_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Features (Model 4 - Enhanced):")
print(importance_df.head(15))

# Metadata features only
metadata_cols = ['unknown', 'action', 'adventure', 'animation', 'children',
                 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                 'film_noir', 'horror', 'musical', 'mystery', 'romance',
                 'sci_fi', 'thriller', 'war', 'western', 'release_year_normalized']
metadata_importance = importance_df[importance_df['feature'].isin(metadata_cols)]

print("\nMetadata Feature Importance:")
print(metadata_importance.sort_values('importance', ascending=False))
print(f"\nTotal metadata contribution: {metadata_importance['importance'].sum()*100:.2f}%")

# 14. Sample predictions comparison
print("\n" + "="*100)
print("SAMPLE PREDICTIONS (First 5 test cases)")
print("="*100)
print(f"{'User':<6} {'Item':<6} {'Actual':<7} {'Model1':<7} {'Model2':<7} {'Model3':<7} {'Model4':<7}")
print("-"*100)

for i in range(min(5, len(y_test_als))):
    idx = valid_test_indices[i]
    user = test_df.loc[idx]['user_id']
    item = test_df.loc[idx]['item_id']
    actual = y_test_als[i]
    pred1 = np.clip(y_pred_als[i], 1, 5)
    pred2 = np.clip(y_pred_xgb_als[i], 1, 5)
    pred3 = np.clip(y_pred_als[i], 1, 5)
    pred4 = np.clip(y_pred_xgb_enhanced[i], 1, 5)

    print(f"{user:<6} {item:<6} {actual:<7} {pred1:<7.2f} {pred2:<7.2f} {pred3:<7.2f} {pred4:<7.2f}")
print("="*100)
