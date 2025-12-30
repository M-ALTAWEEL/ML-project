import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shap

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RM AI - World Happiness", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM STYLING & LOGO ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# App Header with your Logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=150)
with col2:
    st.title("World Happiness Predictor & Analyzer")
    st.markdown("*Powered by RM AI - Leveraging Machine Learning for Global Insights*")

st.divider()

# --- DATA ENGINE (STAGES 2 & 3) ---
@st.cache_data
def get_processed_data():
    # Loading internal files (Make sure these are in your GitHub)
    happy_df = pd.read_csv('world-happiness-report-2021.csv')
    health_df = pd.read_csv('global_health.csv')
    
    # Filter and Clean
    health_2019 = health_df[health_df['Year'] == 2019].copy()
    country_corrections = {
        'United States': 'United States', 'Russia': 'Russian Federation',
        'South Korea': 'Korea, Rep.', 'Taiwan': 'Taiwan Province of China'
    }
    health_2019['Country'] = health_2019['Country'].replace(country_corrections)
    
    merged_df = pd.merge(happy_df, health_2019, left_on='Country name', right_on='Country', how='inner')
    
    # Impute Missing Values
    cols_to_fix = ['Unemployment_Rate', 'Fertility_Rate', 'Suicide_Rate_Percent',
                   'Alcohol_Consumption_Per_Capita', 'Obesity_Rate_Percent']
    for col in cols_to_fix:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    # Encoding & Scaling
    le = LabelEncoder()
    merged_df['Region_Encoded'] = le.fit_transform(merged_df['Regional indicator'])
    
    scaler = MinMaxScaler()
    cols_to_scale = [
        'GDP_Per_Capita', 'Life_Expectancy', 'Alcohol_Consumption_Per_Capita',
        'Obesity_Rate_Percent', 'Unemployment_Rate', 'Fertility_Rate',
        'Freedom to make life choices', 'Perceptions of corruption'
    ]
    merged_df[cols_to_scale] = scaler.fit_transform(merged_df[cols_to_scale])
    
    return merged_df, le, cols_to_scale

df, label_encoder, feature_cols = get_processed_data()

# --- MODEL ENGINE (STAGE 4) ---
@st.cache_resource
def train_final_model(data, features):
    X = data[features + ['Region_Encoded']]
    y = data['Ladder score']
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    return model

final_model = train_final_model(df, feature_cols)

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Happiness Predictor", "Data Analysis (EDA)", "AI Insights (SHAP)", "Clustering (Unsupervised)"])

# --- PAGE 1: PREDICTOR ---
if page == "Happiness Predictor":
    st.header("🔮 Happiness Score Predictor")
    st.info("Adjust the indicators below to see how they impact the predicted happiness score.")
    
    c1, c2 = st.columns(2)
    with c1:
        gdp = st.slider("GDP Per Capita", 0.0, 1.0, float(df['GDP_Per_Capita'].mean()))
        life_exp = st.slider("Life Expectancy", 0.0, 1.0, float(df['Life_Expectancy'].mean()))
        freedom = st.slider("Freedom of Choice", 0.0, 1.0, float(df['Freedom to make life choices'].mean()))
        corruption = st.slider("Perception of Corruption", 0.0, 1.0, float(df['Perceptions of corruption'].mean()))
        region = st.selectbox("Global Region", df['Regional indicator'].unique())

    with c2:
        suicide = st.slider("Suicide Rate (%)", 0.0, 30.0, float(df['Suicide_Rate_Percent'].mean()))
        alcohol = st.slider("Alcohol Consumption", 0.0, 1.0, float(df['Alcohol_Consumption_Per_Capita'].mean()))
        obesity = st.slider("Obesity Rate", 0.0, 1.0, float(df['Obesity_Rate_Percent'].mean()))
        unemployment = st.slider("Unemployment Rate", 0.0, 1.0, float(df['Unemployment_Rate'].mean()))
        fertility = st.slider("Fertility Rate", 0.0, 1.0, float(df['Fertility_Rate'].mean()))

    region_code = list(df['Regional indicator'].unique()).index(region)
    
    if st.button("Calculate Happiness Score", type="primary"):
        input_array = np.array([[gdp, life_exp, alcohol, obesity, unemployment, fertility, freedom, corruption, region_code]])
        res = final_model.predict(input_array)[0]
        
        st.metric("Predicted Ladder Score", f"{res:.2f}")
        if res > 6: st.success("Condition: **High Happiness** 😃")
        elif res > 4: st.warning("Condition: **Moderate Happiness** 😐")
        else: st.error("Condition: **Low Happiness** 😔")

# --- PAGE 2: EDA ---
elif page == "Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    
    tab1, tab2 = st.tabs(["Correlations", "Regional Distributions"])
    
    with tab1:
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[feature_cols + ['Ladder score']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Happiness Distribution by Region")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Ladder score', y='Regional indicator', data=df, palette="Set3")
        st.pyplot(fig)

# --- PAGE 3: SHAP ---
elif page == "AI Insights (SHAP)":
    st.header("🧠 AI Decision Logic (SHAP)")
    st.write("This chart shows which factors the AI values most when calculating happiness.")
    
    X_explain = df[feature_cols + ['Region_Encoded']]
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_explain)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_explain, show=False)
    st.pyplot(fig)

# --- PAGE 4: CLUSTERING ---
elif page == "Clustering (Unsupervised)":
    st.header("🔗 Unsupervised Country Clustering")
    
    num_clusters = st.slider("Select Number of Clusters (K)", 2, 6, 3)
    
    # Use same features as Colab
    feat_unsub = df[feature_cols]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(feat_unsub)
    df['Cluster'] = kmeans.labels_
    
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(feat_unsub)
    df['PCA1'] = pca_res[:,0]
    df['PCA2'] = pca_res[:,1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', s=100)
    st.pyplot(fig)
    
    st.subheader("Countries by Cluster")
    st.dataframe(df[['Country name', 'Cluster', 'Ladder score']].sort_values(by="Cluster"))
