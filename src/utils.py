import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def maps_labels(df, value_maps):
    
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].map(value_maps['PreferredLoginDevice'])
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].map(value_maps['PreferredPaymentMode'])
    df['PreferedOrderCat'] = df['PreferedOrderCat'].map(value_maps['PreferedOrderCat'])

    return df

def encode_features(df):
    
    encoders = {}
    encoder_login_device = LabelEncoder()
    
    df["PreferredLoginDevice"] = encoder_login_device.fit_transform(df["PreferredLoginDevice"])
    encoders["PreferredLoginDevice"] = encoder_login_device
    print('PreferredLoginDevice encoding done')
    encoder_payment_mode = LabelEncoder()
    df["PreferredPaymentMode"] = encoder_payment_mode.fit_transform(df["PreferredPaymentMode"])
    encoders["PreferredPaymentMode"] = encoder_payment_mode
    print('PreferredPaymentMode encoding done')
    
    encoder_order_cat = LabelEncoder()
    df["PreferedOrderCat"] = encoder_order_cat.fit_transform(df["PreferedOrderCat"])
    encoders["PreferedOrderCat"] = encoder_order_cat
    print('PreferedOrderCat encoding done')

    # Save encoders to file
    encoder_file = "encoders.pkl"
    with open(encoder_file, "wb") as file:
        pickle.dump(encoders, file)

    print(f"Encoders saved to {encoder_file}.")
    return df


def fill_missing_values(df, missing_maps):
    for col, method in missing_maps.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        if df[col].isnull().any():
            if method == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                raise ValueError("Method must be either 'mean' or 'mode'.")
    return df

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_encoders(path):
    """Load the pickled encoders from the specified path."""
    with open(path, 'rb') as file:
        encoders = pickle.load(file)
    return encoders

def encode_live_data(df, encoders):
    """Encode the features of new data using the loaded encoders."""
    if "PreferredLoginDevice" in df.columns:
        df["PreferredLoginDevice"] = encoders["PreferredLoginDevice"].transform(df["PreferredLoginDevice"])
    if "PreferredPaymentMode" in df.columns:
        df["PreferredPaymentMode"] = encoders["PreferredPaymentMode"].transform(df["PreferredPaymentMode"])
    if "PreferedOrderCat" in df.columns:
        df["PreferedOrderCat"] = encoders["PreferedOrderCat"].transform(df["PreferedOrderCat"])
    return df