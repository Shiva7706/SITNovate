import streamlit as st
import joblib
import pandas as pd
from gtts import gTTS
import os

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

# Predefined translations
TRANSLATIONS = {
    'hi': {
        'Crop Production Advisor': 'फसल उत्पादन सलाहकार',
        'Crop Year': 'फसल वर्ष',
        'Area (acres)': 'क्षेत्र (एकड़)',
        'Fertilizer (kg/acre)': 'उर्वरक (किग्रा/एकड़)',
        'Pesticide (kg/acre)': 'कीटनाशक (किग्रा/एकड़)',
        'Crop': 'फसल',
        'Season': 'मौसम',
        'State': 'राज्य',
        'Predict Production': 'उत्पादन की भविष्यवाणी करें',
        'Predicted Production': 'अनुमानित उत्पादन',
        'Optimization Recommendations': 'अनुकूलन सिफारिशें',
        'Current Value': 'वर्तमान मूल्य',
        'Recommended Value': 'अनुशंसित मूल्य',
        'Expected Improvement': 'अपेक्षित सुधार',
        'Simulate This Change': 'यह परिवर्तन सिमुलेट करें',
        'New Predicted Production': 'नया अनुमानित उत्पादन',
        'tons': 'टन'
    },

    'ta': {  # Tamil
        'Crop Production Advisor': 'பயிர் உற்பத்தி ஆலோசகர்',
        'Crop Year': 'பயிர் ஆண்டு',
        'Area (acres)': 'பகுதி (ஏக்கர்)',
        'Fertilizer (kg/acre)': 'உரம் (கி.கி./ஏக்கர்)',
        'Pesticide (kg/acre)': 'பூச்சிமருந்து (கி.கி./ஏக்கர்)',
        'Crop': 'பயிர்',
        'Season': 'காலம்',
        'State': 'மாநிலம்',
        'Predict Production': 'உற்பத்தியை கணிக்கவும்',
        'Predicted Production': 'கணிக்கப்பட்ட உற்பத்தி',
        'Optimization Recommendations': 'சிறப்பாக்க பரிந்துரைகள்',
        'Current Value': 'தற்போதைய மதிப்பு',
        'Recommended Value': 'பரிந்துரைக்கப்பட்ட மதிப்பு',
        'Expected Improvement': 'எதிர்பார்க்கப்படும் மேம்பாடு',
        'Simulate This Change': 'இந்த மாற்றத்தை மாதிரிபடுத்தவும்',
        'New Predicted Production': 'புதிய கணிக்கப்பட்ட உற்பத்தி',
        'tons': 'டன்'
    },
    'te': {  # Telugu
        'Crop Production Advisor': 'పంట ఉత్పత్తి సలహాదారు',
        'Crop Year': 'పంట సంవత్సరం',
        'Area (acres)': 'ప్రాంతం (ఎకరాలు)',
        'Fertilizer (kg/acre)': 'ఎరువు (కిలో/ఎకరం)',
        'Pesticide (kg/acre)': 'పురుగుమందు (కిలో/ఎకరం)',
        'Crop': 'పంట',
        'Season': 'కాలం',
        'State': 'రాష్ట్రం',
        'Predict Production': 'ఉత్పత్తిని అంచనా వేయండి',
        'Predicted Production': 'అంచనా ఉత్పత్తి',
        'Optimization Recommendations': 'ఉత్తమీకరణ సిఫారసులు',
        'Current Value': 'ప్రస్తుత విలువ',
        'Recommended Value': 'సిఫారసు చేసిన విలువ',
        'Expected Improvement': 'ఆశించిన మెరుగుదల',
        'Simulate This Change': 'ఈ మార్పును అనుకరించండి',
        'New Predicted Production': 'కొత్త అంచనా ఉత్పత్తి',
        'tons': 'టన్నులు'
    },
    'kn': {  # Kannada
        'Crop Production Advisor': 'ಬೆಳೆ ಉತ್ಪಾದನೆ ಸಲಹೆಗಾರ',
        'Crop Year': 'ಬೆಳೆ ವರ್ಷ',
        'Area (acres)': 'ಪ್ರದೇಶ (ಎಕರ)',
        'Fertilizer (kg/acre)': 'ಗೊಬ್ಬರ (ಕೆಜಿ/ಎಕರ)',
        'Pesticide (kg/acre)': 'ಕೀಟನಾಶಕ (ಕೆಜಿ/ಎಕರ)',
        'Crop': 'ಬೆಳೆ',
        'Season': 'ಋತು',
        'State': 'ರಾಜ್ಯ',
        'Predict Production': 'ಉತ್ಪಾದನೆಯನ್ನು ಅಂದಾಜಿಸಿ',
        'Predicted Production': 'ಅಂದಾಜಿಸಲಾಗಿರುವ ಉತ್ಪಾದನೆ',
        'Optimization Recommendations': 'ಅಪ್ಟಿಮೈಸೇಶನ್ ಶಿಫಾರಸುಗಳು',
        'Current Value': 'ಪ್ರಸ್ತುತ ಮೌಲ್ಯ',
        'Recommended Value': 'ಶಿಫಾರಸು ಮಾಡಿದ ಮೌಲ್ಯ',
        'Expected Improvement': 'ನಿರೀಕ್ಷಿಸಲಾದ ಸುಧಾರಣೆ',
        'Simulate This Change': 'ಈ ಬದಲಾವಣೆಯನ್ನು ಅನುಕರಿಸಿ',
        'New Predicted Production': 'ಹೊಸ ಅಂದಾಜು ಉತ್ಪಾದನೆ',
        'tons': 'ಟನ್'
    },
    'ml': {  # Malayalam
        'Crop Production Advisor': 'കൃഷി ഉത്പാദന ഉപദേഷ്ടാവ്',
        'Crop Year': 'കൃഷിവർഷം',
        'Area (acres)': 'വിസ്തൃതി (ഏക്കർ)',
        'Fertilizer (kg/acre)': 'വളം (കി.ഗ്രാം/ഏക്കർ)',
        'Pesticide (kg/acre)': 'കീടനാശിനി (കി.ഗ്രാം/ഏക്കർ)',
        'Crop': 'വിള',
        'Season': 'സീസൺ',
        'State': 'സംസ്ഥാനം',
        'Predict Production': 'ഉത്പാദനം പ്രവചിക്കുക',
        'Predicted Production': 'പ്രവചിച്ച ഉത്പാദനം',
        'Optimization Recommendations': 'മെച്ചപ്പെടുത്തൽ നിർദ്ദേശങ്ങൾ',
        'Current Value': 'നിലവിലുള്ള മൂല്യം',
        'Recommended Value': 'ശുപാർശചെയ്യുന്ന മൂല്യം',
        'Expected Improvement': 'പ്രതീക്ഷിക്കുന്ന മെച്ചപ്പെടുത്തൽ',
        'Simulate This Change': 'ഈ മാറ്റം സിമുലേറ്റ് ചെയ്യുക',
        'New Predicted Production': 'പുതിയ പ്രവചിച്ച ഉത്പാദനം',
        'tons': 'ടൺ'
    },
    'mr': {  # Marathi
        'Crop Production Advisor': 'पीक उत्पादन सल्लागार',
        'Crop Year': 'पीक वर्ष',
        'Area (acres)': 'क्षेत्र (एकर)',
        'Fertilizer (kg/acre)': 'खत (किलो/एकर)',
        'Pesticide (kg/acre)': 'कीटकनाशक (किलो/एकर)',
        'Crop': 'पीक',
        'Season': 'हंगाम',
        'State': 'राज्य',
        'Predict Production': 'उत्पादनाचा अंदाज करा',
        'Predicted Production': 'अंदाजे उत्पादन',
        'Optimization Recommendations': 'ऑप्टिमायझेशन शिफारशी',
        'Current Value': 'वर्तमान मूल्य',
        'Recommended Value': 'शिफारस केलेले मूल्य',
        'Expected Improvement': 'अपेक्षित सुधारणा',
        'Simulate This Change': 'हा बदल अनुकरण करा',
        'New Predicted Production': 'नवीन अंदाजे उत्पादन',
        'tons': 'टन'
    },
    'bn': {  # Bengali
        'Crop Production Advisor': 'ফসল উৎপাদন পরামর্শদাতা',
        'Crop Year': 'ফসল বর্ষ',
        'Area (acres)': 'এলাকা (একর)',
        'Fertilizer (kg/acre)': 'সার (কেজি/একর)',
        'Pesticide (kg/acre)': 'কীটনাশক (কেজি/একর)',
        'Crop': 'ফসল',
        'Season': 'মৌসুম',
        'State': 'রাজ্য',
        'Predict Production': 'উৎপাদন অনুমান করুন',
        'Predicted Production': 'অনুমানকৃত উৎপাদন',
        'Optimization Recommendations': 'অপ্টিমাইজেশন সুপারিশ',
        'Current Value': 'বর্তমান মান',
        'Recommended Value': 'সুপারিশকৃত মান',
        'Expected Improvement': 'প্রত্যাশিত উন্নতি',
        'Simulate This Change': 'এই পরিবর্তন অনুকরণ করুন',
        'New Predicted Production': 'নতুন অনুমানকৃত উৎপাদন',
        'tons': 'টন'
    },
    'gu': {  # Gujarati
        'Crop Production Advisor': 'પાક ઉત્પાદન સલાહકાર',
        'Crop Year': 'પાક વર્ષ',
        'Area (acres)': 'વિસ્તાર (એકર)',
        'Fertilizer (kg/acre)': 'ખાતર (કિલો/એકર)',
        'Pesticide (kg/acre)': 'કીટનાશક (કિલો/એકર)',
        'Crop': 'પાક',
        'Season': 'ઋતુ',
        'State': 'રાજ્ય',
        'Predict Production': 'ઉત્પાદનનું અનુમાન કરો',
        'Predicted Production': 'અનુમાનિત ઉત્પાદન',
        'Optimization Recommendations': 'ઑપ્ટિમાઇઝેશન સૂચનાઓ',
        'Current Value': 'વર્તમાન કિંમત',
        'Recommended Value': 'શિફારસી કિંમત',
        'Expected Improvement': 'અપેક્ષિત સુધારો',
        'Simulate This Change': 'આ ફેરફારનું અનુકરણ કરો',
        'New Predicted Production': 'નવું અનુમાનિત ઉત્પાદન',
        'tons': 'ટન'
    },
    'pa': {  # Punjabi
        'Crop Production Advisor': 'ਫਸਲ ਉਤਪਾਦਨ ਸਲਾਹਕਾਰ',
        'Crop Year': 'ਫਸਲ ਸਾਲ',
        'Area (acres)': 'ਖੇਤਰਫਲ (ਏਕੜ)',
        'Fertilizer (kg/acre)': 'ਖਾਦ (ਕਿਲੋ/ਏਕੜ)',
        'Pesticide (kg/acre)': 'ਕੀਟਨਾਸ਼ਕ (ਕਿਲੋ/ਏਕੜ)',
        'Crop': 'ਫਸਲ',
        'Season': 'ਮੌਸਮ',
        'State': 'ਰਾਜ',
        'Predict Production': 'ਉਤਪਾਦਨ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰੋ',
        'Predicted Production': 'ਭਵਿੱਖਬਾਣੀ ਕੀਤਾ ਉਤਪਾਦਨ',
        'Optimization Recommendations': 'ਅਪਟੀਮਾਈਜੇਸ਼ਨ ਸੁਝਾਵ',
        'Current Value': 'ਮੌਜੂਦਾ ਮੁੱਲ',
        'Recommended Value': 'ਸਿਫਾਰਸ਼ ਕੀਤੀ ਮੁੱਲ',
        'Expected Improvement': 'ਉਮੀਦ ਕੀਤੀ ਗਈ ਸੁਧਾਰ',
        'Simulate This Change': 'ਇਸ ਬਦਲਾਵ ਦੀ ਅਨੁਕਰਣ ਕਰੋ',
        'New Predicted Production': 'ਨਵਾਂ ਭਵਿੱਖਬਾਣੀ ਕੀਤਾ ਉਤਪਾਦਨ',
        'tons': 'ਟਨ'
    }
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

# Initialize session state
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'original_pred' not in st.session_state:
    st.session_state.original_pred = None

def translate_text(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        return TRANSLATIONS[target_lang].get(text, text)
    except KeyError:
        return text

def text_to_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("output.mp3")
        st.audio("output.mp3")
        os.remove("output.mp3")
    except Exception as e:
        st.warning(f"Text-to-speech failed: {str(e)}")

def generate_dynamic_recommendations(inputs, feature_importance, lang):
    recommendations = []
    
    thresholds = {
        'Pesticide': (50 * (1 + feature_importance['Pesticide']), "reduce"),
        'Fertilizer': (100 * (1 + feature_importance['Fertilizer']), "increase"),
        'Area': (5 * (1 + feature_importance['Area']), "increase")
    }
    
    for feature, (threshold, action) in thresholds.items():
        current_value = inputs[feature]
        
        if action == "reduce" and current_value > threshold:
            recommendations.append({
                'type': feature,
                'action': translate_text(f"Reduce {feature} by 20%", lang),
                'value': current_value * 0.8
            })
        elif action == "increase" and current_value < threshold:
            recommendations.append({
                'type': feature,
                'action': translate_text(f"Increase {feature} by 20%", lang),
                'value': current_value * 1.2
            })
    
    return recommendations

def get_improved_prediction(original_input, recommendations):
    modified = original_input.copy()
    for rec in recommendations:
        modified[rec['type']] = rec['value']
    input_df = pd.DataFrame([modified])
    return model.predict(input_df)[0]

def main():
    # Language selection
    lang = st.sidebar.selectbox("🌐", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x])
    st.session_state.lang = lang
    
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
        st.session_state.original_pred = original_pred
        
        prediction_text = f"{t('Predicted Production')}: {original_pred:.2f} {t('tons')}"
        st.success(prediction_text)
        text_to_speech(prediction_text, lang)
        
        # Generate recommendations
        recommendations = generate_dynamic_recommendations(
            input_data, 
            FEATURE_IMPORTANCE,
            lang
        )
        
        if recommendations:
            st.subheader(t("Optimization Recommendations"))
            for rec in recommendations:
                with st.expander(rec['action']):
                    new_pred = get_improved_prediction(input_data, [rec])
                    diff = new_pred - original_pred
                    
                    st.write(t("Current Value:"), rec['value']/1.2 if 'Reduce' in rec['action'] else rec['value']*0.8)
                    st.write(t("Recommended Value:"), rec['value'])
                    st.write(t("Expected Improvement:"), 
                            f"{diff:.2f} {t('tons')} (+{(diff/original_pred)*100:.1f}%)")
                    
                    if st.button(t("Simulate This Change"), key=rec['action']):
                        new_pred_text = f"{t('New Predicted Production')}: {new_pred:.2f} {t('tons')}"
                        st.success(new_pred_text)
                        text_to_speech(new_pred_text, lang)

if __name__ == "__main__":
    main()