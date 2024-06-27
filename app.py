
pip install xgboost streamlit pandas numpy scikit-learn shap matplotlib

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import streamlit as st

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("python.csv")
    return data

data = load_data()
# Rename the specified columns
data = data.rename(columns={
    'Wide.erythrocyte.volume.distribution': 'Red.blood.cell.distribution.width',
    'X25.Hydroxyvitamin.D': '25.Hydroxy.vitamin.D'
})

# Ensure 'Group' is the target column
target_column = 'Group'
data[target_column] = data[target_column].astype('category')

# Set features and labels
X = data.drop(columns=target_column)
y = data[target_column]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=121, stratify=y)

# Define the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.columns)
    ]
)

# Select top 8 features
selector = SelectKBest(f_classif, k=8)
X_train_selected = selector.fit_transform(preprocessor.fit_transform(X_train), y_train)
X_test_selected = selector.transform(preprocessor.transform(X_test))
selected_features = X.columns[selector.get_support()]

# Retrain XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=121)
model.fit(X_train_selected, y_train)

# SHAP values for interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_selected)

# Streamlit web app
st.title("Predictive Model for ADHD Risk Assessment")

# User inputs for the selected features, initialized to 0
inputs = {}
for feature in selected_features:
    inputs[feature] = st.number_input(feature, value=0.0)

# Prediction button
if st.button("Predict"):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Check for missing columns and add them with default values
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Preprocess the input data
    try:
        input_processed = selector.transform(preprocessor.transform(input_df))
    except ValueError as e:
        st.error(f"Error in preprocessing: {e}")
        st.stop()

    # Make prediction
    prediction_proba = model.predict_proba(input_processed)[:, 1][0]
    st.markdown(f"**<p style='font-weight:bold; color:black;'>Based on feature values, predicted possibility of ADHD is: {prediction_proba:.2%}</p>**", unsafe_allow_html=True)

    # Compute SHAP values for the input
    input_shap_values = explainer.shap_values(input_processed)

    # Plot SHAP force plot
    shap.initjs()
    shap.force_plot(explainer.expected_value, input_shap_values[0], input_processed[0], feature_names=selected_features, matplotlib=True)
    st.pyplot(plt)
