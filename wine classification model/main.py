import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset once
@st.cache(allow_output_mutation=True)
def load_data():
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names

# Load trained model and scaler (assuming you saved them)
@st.cache(allow_output_mutation=True)
def load_model():
    # Replace 'random_forest_model.pkl' and 'scaler.pkl' with your saved files
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Main app
def main():
    st.title("Wine Cultivar Classification - Italian Wine Research Institute")
    st.write("Use the form below to input chemical features and classify the wine cultivar.")

    X, y, feature_names, target_names = load_data()
    model, scaler = load_model()

    st.sidebar.header("Input Chemical Features")
    input_features = {}
    for feature in feature_names:
        min_val = float(np.min(X[:, feature_names.index(feature)]))
        max_val = float(np.max(X[:, feature_names.index(feature)]))
        mean_val = float(np.mean(X[:, feature_names.index(feature)]))
        input_val = st.sidebar.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=0.01
        )
        input_features[feature] = input_val

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_features])
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    pred_class = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    confidence = np.max(pred_proba)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Cultivar:** {target_names[pred_class]}")
    st.write(f"**Prediction Confidence:** {confidence*100:.2f}%")
    st.write("**Class Probabilities:**")
    prob_df = pd.DataFrame({
        'Cultivar': target_names,
        'Probability': pred_proba
    }).sort_values(by='Probability', ascending=False)
    st.table(prob_df)

    # Show feature importance
    st.subheader("Feature Importances (from Random Forest)")
    rf_model = model
    fi = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    st.bar_chart(fi.set_index('feature')['importance'])

    # Visualizations
    st.subheader("Chemical Feature Distributions")
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    for ax, feature in zip(axes.flatten(), feature_names[:4]):
        sns.histplot(X[:, feature_names.index(feature)], ax=ax, kde=True)
        ax.set_title(f"{feature} Distribution")
    st.pyplot(fig)

    st.subheader("PCA 2D Visualization of All Samples")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_scaled = scaler.transform(X)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA 2D Projection of Wine Samples')
    plt.legend(target_names, title='Cultivar')
    st.pyplot()

if __name__ == "__main__":
    main()