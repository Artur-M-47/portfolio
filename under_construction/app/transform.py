
import pandas as pd

def cat_columns_to_category(df, list_cat_columns):

    for col in list_cat_columns:
        df[col] = df[col].astype('category')
    
    return df

def split_data(df, target_col='defaulted', test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    import gc
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

def cat_One_hot_encoding(df,list_cat_columns,list_num_columns,train=False):
    import pickle
    import os

    # One-hot encoding
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
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "train_columns.pkl")
    if train == False :
        with open(data_path , "rb") as f:
            train_columns = pickle.load(f)
        df = df.reindex(columns=train_columns, fill_value=0)
    else :
        print("💾 Saving train columns")
        with open(data_path , "wb") as f:
            pickle.dump(df.columns.tolist(), f)
    return df
