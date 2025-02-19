import streamlit as st
import joblib
import pandas as pd
from gtts import gTTS
import os
import numpy as np

# Set page config FIRST
st.set_page_config(page_title="Crop Production Advisor", layout="wide")

# Load artifacts
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

@st.cache_resource
def load_mappings():
    return joblib.load('frequency_mappings.pkl')

model = load_model()
frequency_mappings = load_mappings()

# Supported languages with their translations
LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi', 
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'pa': 'Punjabi'
}

# Predefined translations (same as previous but extended)
TRANSLATIONS = {
    # ... (keep previous translations and add these new entries)
    'en': {
        'recommendation_template': "Adjust {feature} by {percentage}% to {action}",
        'increase': "increase",
        'decrease': "decrease",
        'current_value': "Current Value",
        'recommended_value': "Recommended Value",
        'expected_improvement': "Expected Improvement",
        # ... rest of previous translations
    },
    'hi': {
        'recommendation_template': "{feature} को {percentage}% {action}",
        'increase': "बढ़ाएँ",
        'decrease': "घटाएँ",
        'current_value': "वर्तमान मूल्य",
        'recommended_value': "अनुशंसित मूल्य",
        'expected_improvement': "अपेक्षित सुधार",
        # ... rest of previous translations
    },
    # Add similar entries for other languages
}

# Feature importance for recommendations
FEATURE_IMPORTANCE = {
    'Pesticide': 0.15,
    'Fertilizer': 0.25,
    'Area': 0.3
}

# State rainfall data
state_rainfall = {
    'Maharashtra': 1200,
    'Karnataka': 900,
    'Tamil Nadu': 1000,
    'Gujarat': 800,
    'Punjab': 700,
    'Uttar Pradesh': 950,
    'Madhya Pradesh': 1100,
    'Bihar': 1050,
    'West Bengal': 1500,
    'Andhra Pradesh': 950
}

def translate_text(text, target_lang):
    return TRANSLATIONS[target_lang].get(text, text)

def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("output.mp3")
        st.audio("output.mp3")
        os.remove("output.mp3")
    except Exception as e:
        st.warning(f"Text-to-speech failed: {str(e)}")

def optimize_feature(input_data, feature, lang):
    original_value = input_data[feature]
    original_pred = model.predict(pd.DataFrame([input_data]))[0]
    
    best_pct = 0  # Initialize with 0% change
    best_value = original_value
    best_pred = original_pred
    
    percentages = np.linspace(-40, 40, 17)  # Test -40% to +40% in 5% steps
    
    for pct in percentages:
        modified = input_data.copy()
        modified[feature] = modified[feature] * (1 + pct/100)
        current_pred = model.predict(pd.DataFrame([modified]))[0]
        
        if current_pred > best_pred:
            best_pred = current_pred
            best_pct = pct
            best_value = modified[feature]

    improvement = best_pred - original_pred
    return best_pct, best_value, improvement

def generate_dynamic_recommendations(input_data, lang):
    recommendations = []
    
    for feature in FEATURE_IMPORTANCE.keys():
        original_value = input_data[feature]
        optimal_pct, optimal_value, improvement = optimize_feature(input_data, feature, lang)
        
        if abs(optimal_pct) < 5:  # Ignore small adjustments
            continue
            
        action = 'increase' if optimal_pct > 0 else 'decrease'
        recommendation = {
            'type': feature,
            'percentage': abs(round(optimal_pct, 1)),
            'action': action,
            'value': optimal_value,
            'improvement': improvement
        }
        recommendations.append(recommendation)
    
    # Sort by improvement potential
    return sorted(recommendations, key=lambda x: x['improvement'], reverse=True)

def main():
    # Language selection
    lang = st.sidebar.selectbox("🌐", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    
    # Translation function
    t = lambda text: translate_text(text, lang)
    
    st.title(t("Crop Production Advisor"))
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            crop_year = st.number_input(t("Crop Year"), min_value=2000, max_value=2030, value=2024)
            area = st.number_input(t("Area (acres)"), min_value=0.1, value=1.0)
            fertilizer = st.number_input(t("Fertilizer (kg/acre)"), min_value=0.0, value=50.0)
            pesticide = st.number_input(t("Pesticide (kg/acre)"), min_value=0.0, value=5.0)
            
        with col2:
            crop = st.selectbox(t("Crop"), options=list(frequency_mappings['Crop'].keys()))
            season = st.selectbox(t("Season"), options=list(frequency_mappings['Season'].keys()))
            state = st.selectbox(t("State"), options=list(frequency_mappings['State'].keys()))
        
        submitted = st.form_submit_button(t("Predict Production"))
    
    if submitted:
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
        
        # Initial prediction
        input_df = pd.DataFrame([input_data])
        original_pred = model.predict(input_df)[0]
        
        prediction_text = f"{t('Predicted Production')}: {original_pred:.2f} {t('tons')}"
        st.success(prediction_text)
        text_to_speech(prediction_text, lang)
        
        # Generate recommendations
        recommendations = generate_dynamic_recommendations(input_data, lang)
        
        if recommendations:
            st.subheader(t("Optimization Recommendations"))
            for rec in recommendations:
                rec_text = t("recommendation_template").format(
                    feature=t(rec['type']),
                    percentage=rec['percentage'],
                    action=t(rec['action'])
                )
                
                with st.expander(rec_text):
                    st.write(f"{t('current_value')}: {input_data[rec['type']]:.2f}")
                    st.write(f"{t('recommended_value')}: {rec['value']:.2f}")
                    st.write(f"{t('expected_improvement')}: {rec['improvement']:.2f} {t('tons')}")
                    
                    if st.button(t("Simulate This Change"), key=rec['type']):
                        modified_data = input_data.copy()
                        modified_data[rec['type']] = rec['value']
                        new_pred = model.predict(pd.DataFrame([modified_data]))[0]
                        new_pred_text = f"{t('New Predicted Production')}: {new_pred:.2f} {t('tons')}"
                        st.success(new_pred_text)
                        text_to_speech(new_pred_text, lang)

if __name__ == "__main__":
    main()