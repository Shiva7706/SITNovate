# KisanMITRA - AI-Driven Crop Yield & Fertilizer Optimization System 🌾

## 🚀 Overview
KisanMITRA is a **Machine Learning-powered Web Application** designed to help farmers **optimize crop yield** by predicting production levels and recommending the **best fertilizers and pesticides** with suitable amounts. The system leverages **local environmental, climatic, and socio-economic factors** to provide actionable insights for farmers.

### 🔥 Key Integrations:
- 🌿 **Crop Yield Prediction** using advanced ML models (**XGBoost**)
- 🧪 **Fertilizer & Pesticide Recommendations** for soil fertility and pest control
- 📍 **Localized Insights** tailored for specific geographic regions (**States**)
- 🎙 **Voice Assistance** for ease of use and accessibility
- 🔠 **Versatality in Language** for ease of use for Farmers

---

## 🌟 Features
✅ **Predict Crop Yield** based on **climate, soil conditions, and socio-economic factors**  
✅ **Personalized Fertilizer & Pesticide Suggestions** with **adjustable dosage inputs**  
✅ **Real-time Environmental Analysis** to dynamically adapt recommendations  
✅ **Voice-Based Assistance** using **GTTS** for **easy accessibility**  
✅ **Scalable Cloud Deployment** to ensure **real-world farmer adoption**  
✅ **Supports 9 Indic Languages** for enhanced accessibility for farmers  
namely - English , Kannada , Hindi , Marathi , Gujarathi , Bengali, Punjabi , Telugu , Tamil , Malyalam
✅ **Clean Interface** to ensure Farmers feel it easy to use
---

## 🛠️ Tech Stack
- **Machine Learning:** XGBoost, Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Deployment:** Streamlit  
- **Model Persistence:** Joblib  
- **Text-to-Speech:** gTTS  
- **Multilingual Support:** NLP techniques for **9 Indic languages**  

---

## 🔗 Access the Application
🌐 **Try it out here:** [KisanMITRA Web App](https://kisanmitra.streamlit.app/) 🚜

---
**Innovation & Creativity** : 
1. Providing recommendation to improve production of crops using only Classical Machine Learning
2. Introducing the interface in 9 different Indic languages without using external API (providing the output in the language favoured by Farmer)
3. Providing Speech output without using external API , used gTTS (supports all 9 Indic languages)

**Technical Complexity** :
1. The Prediction + Recommendation system is completely based on Classical Machine learning
2. Finding suitable data , preprocessing it was a tedious task


## 🔄 Workflow

Below is the workflow diagram for **KisanMITRA**:

```mermaid
flowchart TD
    A["User Inputs:<br>Crop, Region, Environmental Data,<br>Fertilizer & Pesticide Details"]
    B["Data Preprocessing & Feature Extraction"]
    C["Crop Yield Prediction<br>(XGBoost Model)"]
    D["Fertilizer & Pesticide Recommendation<br>(ML Module)"]
    E["Generate Yield Prediction"]
    F["Generate Optimization Recommendations"]
    G["Combine Results"]
    H["Streamlit UI Display"]
    I["Voice Assistance<br>(gTTS)"]
    J["Multilingual Support<br>(9 Indic Languages)"]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I
    H --> J
