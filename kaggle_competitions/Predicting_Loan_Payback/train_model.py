"""
📊 train_model.py - Script for training the Loan Repayment Prediction Model (CatBoost)

This production script is designed to train and save the final 
CatBoost model 
for predicting loan repayment ('loan_paid_back'). It utilizes pre-defined, 
optimized hyperparameters.

Required Input Files:
- data/train.csv: Training data.
- app/transform.py: Module containing transformation functions.

Output:
- list_cat_columns.pkl: Pickle file with categorical column names.
- list_num_columns.pkl: Pickle file with numerical column names.
- app/model.pkl: The saved, trained CatBoost model.
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

# Transformation Module
try:
    import app.transform as transform
except ImportError:
    print("❌ Error: Module 'app.transform' not found. Ensure 'app/transform.py' exists.")
    sys.exit(1)

# --- Paths ---
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "train.csv")
model_output_path = os.path.join(base_dir, "app", "model.pkl")

# --- Loading Training Data ---
try:
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully from: {data_path}")
except Exception as e:
    print(f"❌ Error loading data from {data_path}: {e}")
    sys.exit(1)

df.set_index('id', inplace=True)

target_col = 'loan_paid_back'

# --- Feature Engineering ---
df["calc_ratio"] = df["loan_amount"] / df["annual_income"]

# --- Definicja Kolumn ---

list_num_columns = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',"calc_ratio"]
list_cat_columns = ['gender', 'marital_status', 'education_level', 'employment_status',
                    'loan_purpose', 'grade_subgrade']

# Save column lists for future reference in transformations 

try:
    with open(os.path.join(base_dir, 'app', "list_cat_columns.pkl"), "wb") as f:
        pickle.dump(list_cat_columns, f)
    with open(os.path.join(base_dir, 'app', "list_num_columns.pkl"), "wb") as f:
        pickle.dump(list_num_columns, f)
    print("✅ Column lists saved.")
except Exception as e:
    print(f"❌ Error saving column lists: {e}")
    sys.exit(1)


# --- Data Transformation For Model Traing ---

# Transformation category features to 'category'
df_cat = transform.cat_columns_to_category(df, list_cat_columns)

# Data Splitting 
# CatBoost natively handles categorical features, so no One-Hot Encoding is needed here.
try:
    train_indices, valid_indices, X_train_split, X_valid_split, y_train_split, y_valid_split = transform.split_data(
        df_cat, target_col = target_col
    )
    print(f"✅ Data split: Training ({len(X_train_split)}), Validation ({len(X_valid_split)})")
except Exception as e:
    print(f"❌ Error during data splitting: {e}")
    sys.exit(1)


# CatBoost pools
try:
    X_train_pool = Pool(X_train_split, y_train_split, cat_features=list_cat_columns)
    X_valid_pool = Pool(X_valid_split, y_valid_split, cat_features=list_cat_columns)
except Exception as e:
    print(f"❌ Error creating CatBoost Pool: {e}")
    sys.exit(1)

# --- Model CatBoost Trening ---
def train_catboost_production(X_train_pool, X_valid_pool,y_train, y_valid_split):
    """Trains the optimized CatBoost model and returns the trained model and AUC score."""
    print("\n--- Starting CatBoost Model Training ---")
    
    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / positive_count 

    # Optymalne parametry dla modelu produkcyjnego (z Twojej analizy)
    custom_params = {
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': 0.075,
        'iterations': 5000,
        'early_stopping_rounds': 200,
        'random_seed': 42,
        'verbose': 100, # Log co 100 iterations
        'thread_count': -1, # Use all available cores
        'task_type': 'CPU', # Uncomment to use GPU in a production environment
    }

    model = CatBoostClassifier(**custom_params)

    # Trenowanie
    model.fit(
        X_train_pool,
        eval_set=X_valid_pool,
        verbose=custom_params['verbose'],
        early_stopping_rounds=custom_params['early_stopping_rounds']
    )

    # Ocena końcowa na zestawie walidacyjnym
    p_valid = model.predict_proba(X_valid_pool)[:, 1]
    auc_score = roc_auc_score(y_valid_split, p_valid)
    
    print(f"\n✅ Training complete. Final ROC AUC: {auc_score:.5f}")
    
    return model, auc_score

# Uruchomienie trenowania
final_cat_model, final_auc_score = train_catboost_production(X_train_pool, X_valid_pool, y_train_split, y_valid_split)

# --- Saving the Final Model ---
try:
    with open(model_output_path, "wb") as f:
        pickle.dump(final_cat_model, f)
    print(f"\n💾 Successfully saved the CatBoost model to: {model_output_path}")
    print(f"🎉 Model ready for production with validation AUC: {final_auc_score:.5f}")
except Exception as e:
    print(f"❌ Error saving model to file: {e}")
    sys.exit(1)

def plot_feature_importance(model, feature_dataframe, top_n=10, plot_title="Model Feature Importance"):
    import matplotlib.pyplot as plt
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
    model=final_cat_model,
    feature_dataframe=X_train_split,
    plot_title="Loan Risk Estamation Feature Importance"
)

# --- Cleanup ---
del df, df_cat, X_train_split, X_valid_split, y_train_split, y_valid_split, X_train_pool, X_valid_pool
gc.collect()

sys.exit(0)