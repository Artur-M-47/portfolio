import pandas as pd
import numpy as np
import os
import sys
print("numpy version",np.__version__)
print("pandas version",pd.__version__)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import gc

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "train.csv")

try :
    df = pd.read_csv(data_path)
except:
    print("Zawartość folderu data/:", os.listdir("data"))
    print("Zawartość folderu data/:", os.listdir("data"))
    print(f"❌ Failed to read data from {data_path}")
    sys.exit(1)

df.set_index('id', inplace=True)

print(df.head())
print(df.info())
print(df.isnull().sum())

list_num_columns = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate ']

list_cat_columns = ['gender','marital_status','education_level','employment_status',
                     'loan_purpose','grade_subgrade']

def cat_columns_to_category(df, list_cat_columns):

    for col in list_cat_columns:
        df[col] = df[col].astype('category')
    
    return df

df_cat = cat_columns_to_category(df, list_cat_columns)

def split_data(df, target_col='defaulted', test_size=0.2, random_state=42):
        # --- DATA SPLIT: Train and Validation ---
    X_train_full_data = df.drop([target_col], axis=1)
    y_train_full_data = df[target_col]

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train_full_data, y_train_full_data,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_full_data
    )
    train_indices = X_train_split.index.values
    valid_indices = X_valid_split.index.values

    print(f"Total training data size: {len(df)}")
    print(f"New Train Set size: {len(X_train_split)}")
    print(f"Validation Set size: {len(X_valid_split)}")

    # Free memory
    del y_train_full_data
    gc.collect()
    return train_indices, valid_indices, X_train_split, X_valid_split, y_train_split, y_valid_split

train_indices, valid_indices, X_train_split, X_valid_split, y_train_split, y_valid_split=split_data(df_cat, target_col='loan_paid_back')

def cat_One_hot_encoding(df,list_cat_columns,list_num_columns,columns_after_train_encoded=None):

    train_cols = columns_after_train_encoded

    df = pd.get_dummies(df,columns = list_cat_columns,drop_first=True)
    
    # Find all columns created by one-hot encoding (everything except numeric columns)
    one_hot_encoded_columns = df.columns.difference(list_num_columns)
    
    # Check if one-hot encoded columns contain only 0 or 1 values
    for col in one_hot_encoded_columns:
        if not df[col].dropna().apply(lambda x: x in [0, 1]).all():
            print(f"Non-binary values found in column: {col}")
            print(df[col].unique())

    # Convert one-hot encoded columns to integer type (0 or 1)
    df[one_hot_encoded_columns] = df[one_hot_encoded_columns].astype(int)
    # If training columns are provided, reindex to match them and fill missing columns with 0
    if not train_cols is None:
        df = df.reindex(columns=train_cols, fill_value=0)
    return df

X_train_ohe = cat_One_hot_encoding(X_train_split,list_cat_columns,list_num_columns)
X_valid_ohe = cat_One_hot_encoding(X_valid_split,list_cat_columns,list_num_columns,X_train_ohe.columns)

X_train_ohe = X_train_ohe.fillna(0)
X_valid_ohe = X_valid_ohe.fillna(0)


X_valid_ohe.head()

def train_model(model_type, X_train, y_train, X_valid, y_valid, params=None, results_df=None):
    """
    Trains a selected model and evaluates it using ROC AUC.
    """
    import time
    from datetime import datetime
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    start_time = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")

    # --- Model Selection ---
    if model_type == 'xgboost':
        import xgboost as xgb
        base_model = xgb.XGBClassifier()
    elif model_type == 'lightgbm':
        import lightgbm as lgb
        base_model = lgb.LGBMClassifier()
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # --- Merge with default parameters ---
    if params is not None:
        default_params = base_model.get_params()
        default_params.update(params)
        base_model.set_params(**default_params)

    model = base_model

    # --- Training ---
    model.fit(X_train, y_train)

    # --- Evaluation ---
    p_valid = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, p_valid)
    duration = (time.time() - start_time) / 60

    print(f"\n[{model_type.upper()}] ROC AUC: {auc_score:.5f} | Time: {duration:.2f} min")

    # --- Logging ---
    log_entry = {
        "model_name": model_type,
        "score": round(auc_score, 5),
        "duration_min": round(duration, 2),
        "timestamp": current_time,
        **model.get_params()
    }

    if results_df is None:
        results_df = pd.DataFrame([log_entry])
    else:
        results_df = pd.concat([results_df, pd.DataFrame([log_entry])], ignore_index=True)

    return model, model.get_params(), results_df

# ------------------------------------------------
# Train Models
# ------------------------------------------------
results_df = pd.DataFrame()
# ------------------------------------------------
# RandomForestClassifier

# 🧠 Data Prep Summary
# Random Forest is a tree-based ensemble method that builds multiple decision trees and averages their predictions. 
# Unlike boosting methods, it trains trees independently and is robust to overfitting and noise.

# ✅ 1. Categorical Features Must be encoded — Random Forest does not support categorical variables natively. 
# One-Hot Encoding

# ✅ 2. Scaling Not needed — tree models are scale-invariant. Raw numerical values are fine.

# ✅ 3. Missing Values Not handled automatically — you need to impute missing values before training. 
# Use SimpleImputer or similar preprocessing.
# ------------------------------------------------
model_RFC, model_params_RFC, results_df=train_model("random_forest", X_train_ohe, y_train_split, X_valid_ohe, y_valid_split, params=None, results_df=None)


# ------------------------------------------------
# XGBoost Classifier

# 🧠 Data Prep Summary
# XGBoost is a tree-based ensemble method using gradient boosting — unlike single decision trees, 
# it builds many trees sequentially to correct previous errors and improve accuracy.

# ✅ 1. Categorical Features - One-Hot Encoding (OHE) Required for categorical features — 
# XGBoost doesn’t support them natively.

# ✅ 2. Scaling Not needed — tree models are scale-invariant. Raw numerical values work fine.

# ✅ 3. Missing Values Handled internally (np.nan is supported). No manual imputation required, 
# but it's good to monitor missing data.
# ------------------------------------------------

model_XGB, model_params_XGB, results_df=train_model("xgboost", X_train_ohe, y_train_split, X_valid_ohe, y_valid_split, params=None, results_df=None)

# ------------------------------------------------
# LGBMClassifier

# 🧠 Data Prep Summary
# LightGBM is a tree-based boosting algorithm optimized for speed and memory efficiency. Unlike traditional decision trees, it builds trees leaf-wise for better accuracy and supports large datasets with high performance.

# ✅ 1. Categorical Features Can be passed directly as category dtype — no need for one-hot encoding. LightGBM handles categorical splits natively.

# ✅ 2. Scaling Not needed — tree models are scale-invariant. Raw numerical values work fine.

# ✅ 3. Missing Values Handled internally (np.nan is supported). No manual imputation required, but it's good to monitor missing data.
# ------------------------------------------------
model_LGBM, model_params_LGBM, results_df=train_model("lightgbm", X_train_split, y_train_split, X_valid_split, y_valid_split, params=None, results_df=None)



#  ------------------------------------------------
# CatBoostClassifier
# 🧠 Data Prep & Modeling Summary
# CatBoost is a gradient boosting algorithm based on decision trees, designed for high performance and native support for 
# categorical features. Unlike other boosting methods, it handles categorical data internally and requires minimal preprocessing.

# ✅ 1. Categorical Features - No need for one-hot encoding — CatBoost handles them natively.

# ✅ 2. Scaling Not needed — tree-based models are scale-invariant.

# ✅ 3. Missing Values Handled internally (np.nan is supported), but monitoring is recommended.

# ⚙️ Base Model Configuration & Imbalance Handling
# Cost-Sensitive Learning scale_pos_weight was set as the ratio of negative to positive samples to penalize misclassification 
# of the minority class(positive response), and improve AUC.
# Acceleration & Evaluation Strategy
# The parameter task_type='GPU' was enabled to leverage GPU acceleration, significantly reducing training time.
# Since CatBoost does not support AUC as a loss function, cross-entropy (logloss) was used for training and early stopping.
# AUC was calculated only after training, on the validation set, to evaluate model performance and guide final selection.
# Regularization & Control High iterations (e.g. 15000) combined with early_stopping_rounds=200 helped prevent overfitting and 
# identify the optimal number of trees.

def train_catboost_baseline_auc(
    X_train_pool,
    X_valid_pool,
    # X_test_pool,
    y_valid_split,
    df_model,
    model_name="CatBoost_Base",
    custom_params=None
):
    """
    Trains a CatBoost model using default or user-defined parameters and evaluates performance using ROC AUC.

    Args:
        X_train_pool: CatBoost Pool for training
        X_valid_pool: CatBoost Pool for validation
        X_test_pool: CatBoost Pool for testing
        y_valid_split: True labels for validation set
        df_model: DataFrame to log model results
        model_name: Name to assign to the model
        custom_params: Optional dictionary of CatBoost hyperparameters

    Returns:
        model: trained CatBoost model
        used_params: final parameter dictionary
        df_model: updated log DataFrame
        p_test: predicted probabilities on test set
    """
    import time
    from datetime import datetime
    from sklearn.metrics import roc_auc_score
    from catboost import CatBoostClassifier
    import pandas as pd

    start_time = time.time()
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\n--- Starting CatBoost Training at {current_time} ---")

    # Create base model with default parameters
    model = CatBoostClassifier()

    # Get default parameters from model
    used_params = model.get_params()

    # Update with any custom parameters provided
    if custom_params is not None:
        used_params.update(custom_params)

    # Re-initialize model with updated parameters
    model = CatBoostClassifier(**used_params)

    # Train model
    model.fit(
        X_train_pool,
        eval_set=X_valid_pool,
        verbose=used_params.get('verbose', 100),
        early_stopping_rounds=used_params.get('early_stopping_rounds', None)
    )

    # Evaluate on validation set
    p_valid = model.predict_proba(X_valid_pool)[:, 1]
    auc_score = roc_auc_score(y_valid_split, p_valid)
    print(f"\nValidation ROC AUC Score: {auc_score:.5f}")

    # Predict on test set
    # p_test = model.predict_proba(X_test_pool)[:, 1]
    duration = (time.time() - start_time) / 60
    print(f"Model Training Time: {duration:.2f} minutes")

    # Log results
    log_entry = {
        "model_name": model_name,
        "metric": "ROC AUC",
        "score": round(auc_score, 5),
        "duration_min": round(duration, 2),
        "timestamp": current_time,
        **used_params
    }

    df_model = pd.concat([
        df_model,
        pd.DataFrame([log_entry])
    ], ignore_index=True)

    return model, used_params, df_model, p_test

# Creating CatBoost Data Pools
from catboost import CatBoostClassifier, Pool
X_train_pool = Pool(X_train_split, y_train_split, cat_features=list_cat_columns)
X_valid_pool = Pool(X_valid_split, y_valid_split, cat_features=list_cat_columns)

custom_params = {
    'eval_metric': 'AUC',
    'learning_rate': 0.05
}

cat_clf, cat_params, df_model_results, p_test_cat_clf = train_catboost_baseline_auc(
    X_train_pool, X_valid_pool, y_valid_split,
    df_model=results_df,
    model_name="CatBoost_baseline",
    custom_params=custom_params
)

print("\n=== Model Training Summary ===")
print(results_df)
# Save the best model (based on ROC AUC)
best_model_row = results_df.loc[results_df['score'].idxmax()]
best_model_name = best_model_row['model_name']
print(f"\nBest Model: {best_model_name} with ROC AUC: {best_model_row['score']}")