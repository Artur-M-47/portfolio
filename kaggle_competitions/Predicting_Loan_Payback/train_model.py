"""
📊 train_model
Python Script: Loan Repayment Prediction Model Training and Evaluation

This Python script is designed to train and evaluate multiple classification models
on loan data to predict whether a loan will be paid back (loan_paid_back).
It serves as a benchmark for comparing the performance of different machine learning
algorithms on the same dataset:

⏹️ Random Forest
⏹️ XGBoost
⏹️ LightGBM
⏹️ CatBoost

Data Preprocessing & Transformations:
- Each model received input data transformed according to its requirements.
- Techniques included:
    • Conversion of categorical features to appropriate types
    • One-Hot Encoding for tree-based models that require numerical input
    • Category handling for CatBoost (native categorical support)
    • Ensuring numerical features were properly scaled or cast to float

Evaluation:
- Models were trained and validated on the same dataset split.
- Performance metrics (e.g., accuracy, ROC-AUC) were compared
  to identify the most effective algorithm for loan repayment prediction.

Usage:
    python train_model.py

Output:
    Trained models and evaluation results stored for further analysis.
"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
print("numpy version",np.__version__)
print("pandas version",pd.__version__)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle
import gc

import app.transform as transform

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "train.csv")

try :
    df = pd.read_csv(data_path)
except:
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


cat_col_path = os.path.join(base_dir,'app', "list_cat_columns.pkl")
with open(cat_col_path , "wb") as f:
    pickle.dump(list_cat_columns, f)

num_col_path = os.path.join(base_dir,'app', "list_num_columns.pkl")
with open(num_col_path , "wb") as f:
    pickle.dump(list_num_columns, f)

# Transform categorical columns to 'category' dtype
df_cat = transform.cat_columns_to_category(df, list_cat_columns)

# Split data into training and validation sets
train_indices, valid_indices, X_train_split, X_valid_split, y_train_split, y_valid_split=transform.split_data(df_cat, target_col='loan_paid_back')

# One-Hot Encoding for categorical variables
X_train_ohe = transform.cat_One_hot_encoding(X_train_split,list_cat_columns,list_num_columns,train=True)
X_valid_ohe = transform.cat_One_hot_encoding(X_valid_split,list_cat_columns,list_num_columns,train=False)

X_train_ohe = X_train_ohe.fillna(0)
X_valid_ohe = X_valid_ohe.fillna(0)


X_valid_ohe.head()

# ------------------------------------------------
# Train Models
# ------------------------------------------------
# results_df to log model results

results_df = pd.DataFrame()


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

'''
------------------------------------------------
RandomForestClassifier

🧠 Data Prep Summary
Random Forest is a tree-based ensemble method that builds multiple decision trees and averages their predictions. 
Unlike boosting methods, it trains trees independently and is robust to overfitting and noise.

⏹️ 1. Categorical Features Must be encoded — Random Forest does not support categorical variables natively. 
One-Hot Encoding

⏹️ 2. Scaling Not needed — tree models are scale-invariant. Raw numerical values are fine.

⏹️ 3. Missing Values Not handled automatically — you need to impute missing values before training. 
Use SimpleImputer or similar preprocessing.
# ------------------------------------------------'''
model_RFC, model_params_RFC, results_df=train_model("random_forest", X_train_ohe, y_train_split, X_valid_ohe, y_valid_split, params=None, results_df=results_df)

'''
------------------------------------------------
XGBoost Classifier

🧠 Data Prep Summary
XGBoost is a tree-based ensemble method using gradient boosting — unlike single decision trees, 
it builds many trees sequentially to correct previous errors and improve accuracy.

⏹️ 1. Categorical Features - One-Hot Encoding (OHE) Required for categorical features — 
XGBoost doesn’t support them natively.

⏹️ 2. Scaling Not needed — tree models are scale-invariant. Raw numerical values work fine.

⏹️ 3. Missing Values Handled internally (np.nan is supported). No manual imputation required, 
but it's good to monitor missing data.
------------------------------------------------'''

model_XGB, model_params_XGB, results_df=train_model("xgboost", X_train_ohe, y_train_split, X_valid_ohe, y_valid_split, params=None, results_df=results_df)
'''
------------------------------------------------
LGBMClassifier

🧠 Data Prep Summary
LightGBM is a tree-based boosting algorithm optimized for speed and memory efficiency. Unlike traditional decision trees, it builds trees leaf-wise for better accuracy and supports large datasets with high performance.

⏹️ 1. Categorical Features Can be passed directly as category dtype — no need for one-hot encoding. LightGBM handles categorical splits natively.

⏹️ 2. Scaling Not needed — tree models are scale-invariant. Raw numerical values work fine.

⏹️ 3. Missing Values Handled internally (np.nan is supported). No manual imputation required, but it's good to monitor missing data.
------------------------------------------------'''
model_LGBM, model_params_LGBM, results_df=train_model("lightgbm", X_train_split, y_train_split, X_valid_split, y_valid_split, params=None, results_df= results_df)


'''
------------------------------------------------
CatBoostClassifier
🧠 Data Prep & Modeling Summary
CatBoost is a gradient boosting algorithm based on decision trees, designed for high performance and native support for 
categorical features. Unlike other boosting methods, it handles categorical data internally and requires minimal preprocessing.

⏹️ 1. Categorical Features - No need for one-hot encoding — CatBoost handles them natively.

⏹️ 2. Scaling Not needed — tree-based models are scale-invariant.

⏹️ 3. Missing Values Handled internally (np.nan is supported), but monitoring is recommended.

⚙️ Base Model Configuration & Imbalance Handling
Cost-Sensitive Learning scale_pos_weight was set as the ratio of negative to positive samples to penalize misclassification 
of the minority class(positive response), and improve AUC.
Acceleration & Evaluation Strategy
The parameter task_type='GPU' was enabled to leverage GPU acceleration, significantly reducing training time.
Since CatBoost does not support AUC as a loss function, cross-entropy (logloss) was used for training and early stopping.
AUC was calculated only after training, on the validation set, to evaluate model performance and guide final selection.
Regularization & Control High iterations (e.g. 15000) combined with early_stopping_rounds=200 helped prevent overfitting and 
identify the optimal number of trees.
------------------------------------------------'''
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

    return model, used_params, df_model

# Creating CatBoost Data Pools
from catboost import CatBoostClassifier, Pool
X_train_pool = Pool(X_train_split, y_train_split, cat_features=list_cat_columns)
X_valid_pool = Pool(X_valid_split, y_valid_split, cat_features=list_cat_columns)

custom_params = {
    'eval_metric': 'AUC',
    'learning_rate': 0.07,
    'iterations': 5000,
    'early_stopping_rounds': 200
}

cat_clf, cat_params, results_df = train_catboost_baseline_auc(
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

# Ścieżka do zapisu
model_path = os.path.join(os.path.dirname(__file__),"app","model.pkl")

# Zapis modelu
with open(model_path, "wb") as f:
    pickle.dump(cat_clf, f)

print(f"💾 Model zapisany do: {model_path}")


def plot_feature_importance(model, feature_dataframe, top_n=10, plot_title="Model Feature Importance"):
    """
    Retrieves, sorts, and visualizes the feature importance from a trained CatBoost model.

    Args:
        model (CatBoostClassifier): The trained CatBoost model object.
        feature_dataframe (pd.DataFrame): The DataFrame containing the features used for training.
        top_n (int): The number of top features to display.
        plot_title (str): The title for the resulting plot.

    Returns:
        pd.DataFrame: A DataFrame of sorted feature importances.
    """
    
    if not hasattr(model, 'get_feature_importance'):
        print("Error: The provided object does not have a 'get_feature_importance' method.")
        return pd.DataFrame()

    try:
        # 1. Get Feature Importances
        feature_importances = model.get_feature_importance()
        feature_names = feature_dataframe.columns.tolist()

        # 2. Create and Sort DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Select top N features
        top_features = importance_df.head(top_n)

        print(f"\n--- TOP {top_n} {plot_title} ---\n")
        print(top_features)

        # 3. Visualization
        plt.figure(figsize=(10, top_n / 2)) # Dynamic height based on N
        plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
        plt.xlabel("Feature Importance Value")
        plt.ylabel("Feature")
        plt.title(plot_title)
        plt.gca().invert_yaxis() # Invert y-axis for better readability
        
        # 💾 Save the plot
        base_dir = os.path.dirname(__file__)
        image_path = os.path.join(base_dir, "images", "Feature_Importance.png")
        plt.savefig(image_path, bbox_inches='tight')
        
        plt.show() # 

        return importance_df

    except Exception as e:
        print(f"An error occurred during feature importance processing: {e}")
        return pd.DataFrame()
    
loan_importance_df = plot_feature_importance(
    model=cat_clf,
    feature_dataframe=X_train_split,
    plot_title="Loan Risk Estamation Feature Importance"
)