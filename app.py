import streamlit as st
import joblib
import pandas as pd
from googletrans import Translator

# Set page config FIRST
st.set_page_config(page_title="Crop Production Advisor", layout="wide")

# Load artifacts with error handling
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgb_model.pkl')
    except FileNotFoundError:
        st.error("Model file (xgb_model.pkl) not found. Please ensure it exists in the same directory.")
        st.stop()

@st.cache_resource
def load_mappings():
    try:
        return joblib.load('frequency_mappings.pkl')
    except FileNotFoundError:
        st.error("Frequency mapping file (frequency_mappings.pkl) not found. Please ensure it exists in the same directory.")
        st.stop()

model = load_model()
frequency_mappings = load_mappings()

# Load state rainfall data (create this as a dictionary)
state_rainfall = {
    "Punjab": 1200,
    "Maharashtra": 900,
    "Karnataka": 850,
    # Add all states with their average rainfall
}

# Supported languages
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'te': 'Telugu',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'ur': 'Urdu',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam'
}

# Feature importance data
FEATURE_IMPORTANCE = {
    'Pesticide': 0.393632,
    'Area': 0.319622,
    'Fertilizer': 0.109213,
    'Crop': 0.068710,
    'State': 0.047579,
    'Season': 0.034890,
    'Annual_Rainfall': 0.015027,
    'Crop_Year': 0.011328
}

# Initialize translator
translator = Translator()

def translate(text, dest_lang):
    try:
        return translator.translate(text, dest=dest_lang).text
    except:
        return text  # Fallback to original text

def get_recommendations(inputs):
    recommendations = []
    
    # Pesticide recommendation
    if inputs['Pesticide'] > 50:
        recommendations.append("Reduce pesticide usage by 20% for better soil health")
    
    # Area recommendation
    if inputs['Area'] < 5:
        recommendations.append("Consider increasing cultivation area for better yield")
    
    # Fertilizer recommendation
    if inputs['Fertilizer'] < 100:
        recommendations.append("Increase fertilizer usage by 15% for optimal growth")
    
    return recommendations

# Main app
def main():
    # Language selection
    lang = st.sidebar.selectbox("🌐 Select Language", options=list(LANGUAGES.values()))
    lang_code = [k for k, v in LANGUAGES.items() if v == lang][0]

    # Translate common terms
    t = lambda text: translate(text, lang_code)
    
    st.title(t("Crop Production Advisor"))
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            crop_year = st.number_input(t("Crop Year"), min_value=2000, max_value=2030)
            area = st.number_input(t("Area (acres)"), min_value=0.1, max_value=1000.0)
            fertilizer = st.number_input(t("Fertilizer (kg/acre)"), min_value=0.0)
            pesticide = st.number_input(t("Pesticide (kg/acre)"), min_value=0.0)
            
        with col2:
            crop = st.selectbox(t("Crop"), options=list(frequency_mappings['Crop'].keys()))
            season = st.selectbox(t("Season"), options=list(frequency_mappings['Season'].keys()))
            state = st.selectbox(t("State"), options=list(frequency_mappings['State'].keys()))
        
        submitted = st.form_submit_button(t("Predict Production"))
    
    if submitted:
        # Prepare input data
        input_data = {
            'Crop': frequency_mappings['Crop'][crop],
            'Crop_Year': crop_year,
            'Season': frequency_mappings['Season'][season],
            'State': frequency_mappings['State'][state],
            'Area': area,
            'Annual_Rainfall': state_rainfall.get(state, 1000),
            'Fertilizer': fertilizer,
            'Pesticide': pesticide,
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data], columns=[
            'Crop','Crop_Year','Season', 'State','Area', 'Annual_Rainfall',
            'Fertilizer', 'Pesticide'
  
        ])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Generate recommendations
        recommendations = get_recommendations({
            'Pesticide': pesticide,
            'Area': area,
            'Fertilizer': fertilizer
        })
        
        # Display results
        st.success(t(f"Predicted Production: {prediction:.2f} tons"))
        
        if recommendations:
            st.subheader(t("Recommendations"))
            for rec in recommendations:
                st.info(f"✅ {t(rec)}")
        
        # Feature importance visualization
        st.subheader(t("Key Factors Affecting Production"))
        cols = st.columns(len(FEATURE_IMPORTANCE))
        sorted_features = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features):
            cols[i].metric(label=t(feature), value=f"{importance*100:.1f}%")

if __name__ == "__main__":
    main()