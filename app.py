import streamlit as st
import joblib
import pandas as pd
from gtts import gTTS
import os
import numpy as np

st.set_page_config(page_title="Kisan MITRA", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

@st.cache_resource
def load_mappings():
    return joblib.load('frequency_mappings.pkl')

model = load_model()
frequency_mappings = load_mappings()

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

# Updated translations with new labels for fertilizer and pesticide names
TRANSLATIONS = {
    'en': {
        'Kisan MITRA': 'Kisan MITRA',
        'Crop Year': 'Crop Year',
        'Area (acres)': 'Area (acres)',
        'Fertilizer': 'Fertilizer Name',
        'Pesticide': 'Pesticide Name',
        'Crop': 'Crop',
        'Season': 'Season',
        'State': 'State',
        'Predict Production': 'Predict Production',
        'Predicted Production': 'Predicted Production',
        'New Predicted Production': 'New Predicted Production',
        'tons': 'tons',
        'recommendation_template': "Adjust {feature} by {percentage}% to {action}",
        'increase': "increase",
        'decrease': "decrease",
        'current_value': "Current Value",
        'recommended_value': "Recommended Value",
        'Simulate This Change': 'Simulate This Change'
    },
    'hi': {
         'Kisan MITRA': 'फसल उत्पादन सलाहकार',
         'Crop Year': 'फसल वर्ष',
         'Area (acres)': 'क्षेत्र (एकड़)',
         'Fertilizer': 'उर्वरक का नाम',
         'Pesticide': 'कीटनाशक का नाम',
         'Crop': 'फसल',
         'Season': 'मौसम',
         'State': 'राज्य',
         'Predict Production': 'उत्पादन की भविष्यवाणी करें',
         'Predicted Production': 'अनुमानित उत्पादन',
         'New Predicted Production': 'नया अनुमानित उत्पादन',
         'tons': 'टन',
         'recommendation_template': "{feature} को {percentage}% {action}",
         'increase': "बढ़ाएँ",
         'decrease': "घटाएँ",
         'current_value': "वर्तमान मूल्य",
         'recommended_value': "अनुशंसित मूल्य",
         'Simulate This Change': 'इस परिवर्तन को सिमुलेट करें'
    },
    # ... add other languages similarly
    'kn': {
    'Kisan MITRA': 'ಕಿಸಾನ್ ಮಿತ್ರ',
    'Crop Year': 'ಬೆಳೆ ವರ್ಷ',
    'Area (acres)': 'ಪ್ರದೇಶ (ಎಕರ)',
    'Fertilizer': 'ಗೊಬ್ಬರದ ಹೆಸರು',
    'Pesticide': 'ಕೀಟನಾಶಕದ ಹೆಸರು',
    'Crop': 'ಬೆಳೆ',
    'Season': 'ಋತು',
    'State': 'ರಾಜ್ಯ',
    'Predict Production': 'ಉತ್ಪಾದನೆಯನ್ನು ಅಂದಾಜಿಸಿ',
    'Predicted Production': 'ಅಂದಾಜಿಸಲಾಗಿರುವ ಉತ್ಪಾದನೆ',
    'New Predicted Production': 'ಹೊಸ ಅಂದಾಜು ಉತ್ಪಾದನೆ',
    'tons': 'ಟನ್',
    'recommendation_template': "{feature} ಅನ್ನು {percentage}% {action}",
    'increase': "ಹೆಚ್ಚಿಸಿ",
    'decrease': "ಕಡಿಮೆಮಾಡಿ",
    'current_value': "ಪ್ರಸ್ತುತ ಮೌಲ್ಯ",
    'recommended_value': "ಶಿಫಾರಸು ಮಾಡಿದ ಮೌಲ್ಯ",
    'Simulate This Change': 'ಈ ಬದಲಾವಣೆಯನ್ನು ಅನುಕರಿಸಿ'
},

'mr': {
    'Kisan MITRA': 'किसान मित्र',
    'Crop Year': 'पीक वर्ष',
    'Area (acres)': 'क्षेत्र (एकर)',
    'Fertilizer': 'खताचे नाव',
    'Pesticide': 'कीटकनाशकाचे नाव',
    'Crop': 'पीक',
    'Season': 'हंगाम',
    'State': 'राज्य',
    'Predict Production': 'उत्पादनाचा अंदाज करा',
    'Predicted Production': 'अंदाजे उत्पादन',
    'New Predicted Production': 'नवीन अंदाजे उत्पादन',
    'tons': 'टन',
    'recommendation_template': "{feature} ला {percentage}% {action}",
    'increase': "वाढवा",
    'decrease': "कमी करा",
    'current_value': "वर्तमान मूल्य",
    'recommended_value': "शिफारस केलेले मूल्य",
    'Simulate This Change': 'हा बदल अनुकरण करा'
},

'gu': {
    'Kisan MITRA': 'કિસાન મિત્ર',
    'Crop Year': 'પાક વર્ષ',
    'Area (acres)': 'વિસ્તાર (એકર)',
    'Fertilizer': 'ખાતરનું નામ',
    'Pesticide': 'કીટનાશકનું નામ',
    'Crop': 'પાક',
    'Season': 'ઋતુ',
    'State': 'રાજ્ય',
    'Predict Production': 'ઉત્પાદનનું અનુમાન કરો',
    'Predicted Production': 'અનુમાનિત ઉત્પાદન',
    'New Predicted Production': 'નવું અનુમાનિત ઉત્પાદન',
    'tons': 'ટન',
    'recommendation_template': "{feature} ને {percentage}% {action}",
    'increase': "વધારો",
    'decrease': "ઘટાવો",
    'current_value': "વર્તમાન કિંમત",
    'recommended_value': "શિફારસી કિંમત",
    'Simulate This Change': 'આ ફેરફારનું અનુકરણ કરો'
},

'bn': {
    'Kisan MITRA': 'কিসান মিত্র',
    'Crop Year': 'ফসল বর্ষ',
    'Area (acres)': 'এলাকা (একর)',
    'Fertilizer': 'সারের নাম',
    'Pesticide': 'কীটনাশকের নাম',
    'Crop': 'ফসল',
    'Season': 'মৌসুম',
    'State': 'রাজ্য',
    'Predict Production': 'উৎপাদন অনুমান করুন',
    'Predicted Production': 'অনুমানকৃত উৎপাদন',
    'New Predicted Production': 'নতুন অনুমানকৃত উৎপাদন',
    'tons': 'টন',
    'recommendation_template': "{feature} কে {percentage}% {action}",
    'increase': "বৃদ্ধি করুন",
    'decrease': "কমান",
    'current_value': "বর্তমান মান",
    'recommended_value': "সুপারিশকৃত মান",
    'Simulate This Change': "এই পরিবর্তন অনুকরণ করুন"
},

'pa': {
    'Kisan MITRA': 'ਕਿਸਾਨ ਮਿਤ੍ਰ',
    'Crop Year': 'ਫਸਲ ਸਾਲ',
    'Area (acres)': 'ਖੇਤਰਫਲ (ਏਕੜ)',
    'Fertilizer': 'ਖਾਦ ਦਾ ਨਾਮ',
    'Pesticide': 'ਕੀਟਨਾਸ਼ਕ ਦਾ ਨਾਮ',
    'Crop': 'ਫਸਲ',
    'Season': 'ਮੌਸਮ',
    'State': 'ਰਾਜ',
    'Predict Production': 'ਉਤਪਾਦਨ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰੋ',
    'Predicted Production': 'ਭਵਿੱਖਬਾਣੀ ਕੀਤਾ ਉਤਪਾਦਨ',
    'New Predicted Production': 'ਨਵਾਂ ਭਵਿੱਖਬਾਣੀ ਕੀਤਾ ਉਤਪਾਦਨ',
    'tons': 'ਟਨ',
    'recommendation_template': "{feature} ਨੂੰ {percentage}% {action}",
    'increase': "ਵਧਾਓ",
    'decrease': "ਘਟਾਓ",
    'current_value': "ਮੌਜੂਦਾ ਮੁੱਲ",
    'recommended_value': "ਸਿਫਾਰਸ਼ ਕੀਤੀ ਮੁੱਲ",
    'Simulate This Change': "ਇਸ ਬਦਲਾਵ ਦੀ ਅਨੁਕਰਣ ਕਰੋ"
},

'te': {
    'Kisan MITRA': 'కిసాన్ మిత్ర',
    'Crop Year': 'పంట సంవత్సరం',
    'Area (acres)': 'ప్రాంతం (ఎకరాలు)',
    'Fertilizer': 'ఎరువుల పేరు',
    'Pesticide': 'పురుగుమందు పేరు',
    'Crop': 'పంట',
    'Season': 'కాలం',
    'State': 'రాష్ట్రం',
    'Predict Production': 'ఉత్పత్తిని అంచనా వేయండి',
    'Predicted Production': 'అంచనా ఉత్పత్తి',
    'New Predicted Production': 'కొత్త అంచనా ఉత్పత్తి',
    'tons': 'టన్నులు',
    'recommendation_template': "{feature} ను {percentage}% {action}",
    'increase': "పెంచండి",
    'decrease': "తగ్గించండి",
    'current_value': "ప్రస్తుత విలువ",
    'recommended_value': "సిఫారసు చేసిన విలువ",
    'Simulate This Change': "ఈ మార్పును అనుకరించండి"
},

'ta': {
    'Kisan MITRA': 'கிசான் மித்ரா',
    'Crop Year': 'பயிர் ஆண்டு',
    'Area (acres)': 'பகுதி (ஏக்கர்)',
    'Fertilizer': 'உரத்தின் பெயர்',
    'Pesticide': 'பூச்சிமருந்தின் பெயர்',
    'Crop': 'பயிர்',
    'Season': 'காலம்',
    'State': 'மாநிலம்',
    'Predict Production': 'உற்பத்தியை கணிக்கவும்',
    'Predicted Production': 'கணிக்கப்பட்ட உற்பத்தி',
    'New Predicted Production': 'புதிய கணிக்கப்பட்ட உற்பத்தி',
    'tons': 'டன்',
    'recommendation_template': "{feature} ஐ {percentage}% {action}",
    'increase': "பெருக்கவும்",
    'decrease': "குறைக்கவும்",
    'current_value': "தற்போதைய மதிப்பு",
    'recommended_value': "பரிந்துரைக்கப்பட்ட மதிப்பு",
    'Simulate This Change': "இந்த மாற்றத்தை சிமுலேட் செய்யவும்"
},

'ml': {
    'Kisan MITRA': 'കിസാൻ മിത്ര',
    'Crop Year': 'കൃഷിവർഷം',
    'Area (acres)': 'വിസ്തൃതി (ഏക്കർ)',
    'Fertilizer': 'വളത്തിന്റെ പേര്',
    'Pesticide': 'കീടനാശിനിയുടെ പേര്',
    'Crop': 'വിള',
    'Season': 'സീസൺ',
    'State': 'സംസ്ഥാനം',
    'Predict Production': 'ഉത്പാദനം പ്രവചിക്കുക',
    'Predicted Production': 'പ്രവചിച്ച ഉത്പാദനം',
    'New Predicted Production': 'പുതിയ പ്രവചിച്ച ഉത്പാദനം',
    'tons': 'ടൺ',
    'recommendation_template': "{feature} യെ {percentage}% {action}",
    'increase': "കൂട്ടുക",
    'decrease': "കുറയ്ക്കുക",
    'current_value': "നിലവിലുള്ള മൂല്യം",
    'recommended_value': "ശുപാർശചെയ്യുന്ന മൂല്യം",
    'Simulate This Change': "ഈ മാറ്റം സിമുലേറ്റ് ചെയ്യുക"
}

}

# Only include Fertilizer and Pesticide in the recommendations
FEATURE_IMPORTANCE = {
    'Pesticide': 0.15,
    'Fertilizer': 0.25
}

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

# Crop-specific options for fertilizer and pesticide names
crop_fertilizers = {
    'Wheat': ['Urea', 'DAP', 'Potash'],
    'Rice': ['Urea', 'Ammonium Sulfate', 'Superphosphate'],
    'Maize': ['Urea', 'NPK', 'Compost']
}
crop_pesticides = {
    'Wheat': ['Chlorpyrifos', 'Lambda-cyhalothrin'],
    'Rice': ['Imidacloprid', 'Cypermethrin'],
    'Maize': ['Spinosad', 'Bifenthrin']
}
fertilizer_defaults = {
    'Urea': 50.0,
    'DAP': 60.0,
    'Potash': 55.0,
    'Ammonium Sulfate': 45.0,
    'Superphosphate': 40.0,
    'NPK': 50.0,
    'Compost': 30.0
}
pesticide_defaults = {
    'Chlorpyrifos': 5.0,
    'Lambda-cyhalothrin': 4.0,
    'Imidacloprid': 3.5,
    'Cypermethrin': 4.5,
    'Spinosad': 2.5,
    'Bifenthrin': 3.0
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

def generate_dynamic_recommendations(input_data, lang):
    recommendations = []
    for feature in FEATURE_IMPORTANCE.keys():
        try:
            optimal_pct, optimal_value, improvement = optimize_feature(input_data, feature, lang)
            if improvement > 0.001:
                action = 'increase' if optimal_pct > 0 else 'decrease'
                recommendations.append({
                    'type': feature,
                    'percentage': abs(round(optimal_pct, 1)),
                    'action': action,
                    'value': optimal_value,
                    'improvement': improvement
                })
        except Exception as e:
            st.error(f"Error processing {feature}: {str(e)}")
            continue
    return sorted(recommendations, key=lambda x: x['improvement'], reverse=True)

def optimize_feature(input_data, feature, lang):
    original_value = input_data[feature]
    input_df = pd.DataFrame([input_data])
    original_pred = model.predict(input_df)[0]
    
    best_pct = 0
    best_value = original_value
    best_pred = original_pred
    percentages = np.linspace(-40, 40, 17)
    
    for pct in percentages:
        modified = input_data.copy()
        modified[feature] = modified[feature] * (1 + pct/100)
        modified_df = pd.DataFrame([modified])[input_df.columns]
        try:
            current_pred = model.predict(modified_df)[0]
            if current_pred > best_pred:
                best_pred = current_pred
                best_pct = pct
                best_value = modified[feature]
        except Exception as e:
            st.error(f"Prediction failed for {feature} {pct}%: {str(e)}")
            continue
    improvement = best_pred - original_pred
    return best_pct, best_value, improvement

def main():
    lang = st.sidebar.selectbox("🌐", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    t = lambda text: translate_text(text, lang)
    
    st.title(t("Crop Production Advisor"))
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            crop_year = st.number_input(t("Crop Year"), min_value=2000, max_value=2030, value=2024)
            area = st.number_input(t("Area (acres)"), min_value=0.1, value=1.0)
            
        with col2:
            crop = st.selectbox(t("Crop"), options=list(frequency_mappings['Crop'].keys()))
            season = st.selectbox(t("Season"), options=list(frequency_mappings['Season'].keys()))
            state = st.selectbox(t("State"), options=list(frequency_mappings['State'].keys()))
        
        # Dropdowns for fertilizer and pesticide names based on selected crop
        fertilizers = crop_fertilizers.get(crop, list(fertilizer_defaults.keys()))
        pesticide_options = crop_pesticides.get(crop, list(pesticide_defaults.keys()))
        fertilizer_name = st.selectbox(t("Fertilizer"), options=fertilizers)
        pesticide_name = st.selectbox(t("Pesticide"), options=pesticide_options)
        
        submitted = st.form_submit_button(t("Predict Production"))
    
    if submitted:
        # Use default numeric values based on the selected names
        fertilizer_value = fertilizer_defaults.get(fertilizer_name, 50.0)
        pesticide_value = pesticide_defaults.get(pesticide_name, 5.0)
        
        input_data = {
            'Crop': frequency_mappings['Crop'][crop],
            'Crop_Year': crop_year,
            'Season': frequency_mappings['Season'][season],
            'State': frequency_mappings['State'][state],
            'Area': area,
            'Annual_Rainfall': state_rainfall.get(state, 1000),
            'Fertilizer': fertilizer_value,
            'Pesticide': pesticide_value,
        }
        
        expected_columns = [
            'Crop', 'Crop_Year', 'Season', 'State',
            'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'
        ]
        
        input_df = pd.DataFrame([input_data])[expected_columns]
        st.write("Final Input Data:")
        st.write(input_df)
        
        try:
            original_pred = model.predict(input_df)[0]
            prediction_text = f"{t('Predicted Production')}: {original_pred:.2f} {t('tons')}"
            st.success(prediction_text)
            text_to_speech(prediction_text, lang)
            
            recommendations = generate_dynamic_recommendations(input_data, lang)
            
            if not recommendations:
                st.warning("Current inputs are already optimal. Try adjusting different parameters.")
            else:
                st.subheader(t("Optimization Recommendations"))
                for rec in recommendations:
                    rec_text = t("recommendation_template").format(
                        feature=t(rec['type']),
                        percentage=rec['percentage'],
                        action=t(rec['action'])
                    )
                    with st.expander(rec_text):
                        if rec['type'] in ['Fertilizer', 'Pesticide']:
                            current_val = input_data[rec['type']]
                            recommended_val = rec['value']
                        st.write(f"{t('current_value')}: {current_val:.2f}")
                        st.write(f"{t('recommended_value')}: {recommended_val:.2f}")
                        
                        if st.button(t("Simulate This Change"), key=rec['type']):
                            modified_data = input_data.copy()
                            modified_data[rec['type']] = recommended_val
                            modified_df = pd.DataFrame([modified_data])[expected_columns]
                            new_pred = model.predict(modified_df)[0]
                            new_pred_text = f"{t('New Predicted Production')}: {new_pred:.2f} {t('tons')}"
                            st.success(new_pred_text)
                            text_to_speech(new_pred_text, lang)
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Debug - Model input columns:", input_df.columns.tolist())
            st.write("Debug - Input data:", input_df)

if __name__ == "__main__":
    main()
