import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import streamlit as st

@st.cache_data
def load_data():
    data = pd.read_csv("python.csv")
    data = data.rename(columns={
        'Wide.erythrocyte.volume.distribution': 'Red.blood.cell.distribution.width',
        'X25.Hydroxyvitamin.D': '25.Hydroxy.vitamin.D'
    })
    return data

data = load_data()

target_column = 'Group'
data[target_column] = data[target_column].astype('category')

X = data.drop(columns=target_column)
y = data[target_column]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.columns)
    ]
)

selector = SelectKBest(f_classif, k=8)
X_selected = selector.fit_transform(preprocessor.fit_transform(X), y)
selected_features = X.columns[selector.get_support()]

@st.cache_data
def train_model(X_selected, y):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', seed=121)
    model.fit(X_selected, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_selected)
    return model, explainer, shap_values

model, explainer, shap_values = train_model(X_selected, y)

st.title("Predictive Model for ADHD Risk Assessment")

inputs = {}
for feature in selected_features:
    inputs[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    try:
        input_processed = selector.transform(preprocessor.transform(input_df))
    except ValueError as e:
        st.error(f"Error in preprocessing: {e}")
        st.stop()

    prediction_proba = model.predict_proba(input_processed)[:, 1][0]
    st.markdown(f"**<p style='font-weight:bold; color:black;'>Based on feature values, predicted possibility of ADHD is: {prediction_proba:.2%}</p>**", unsafe_allow_html=True)

    input_shap_values = explainer.shap_values(input_processed)

    st_shap = st._legacy_shap
    st_shap.force_plot(explainer.expected_value, input_shap_values[0], input_processed[0], feature_names=selected_features, matplotlib=True)
