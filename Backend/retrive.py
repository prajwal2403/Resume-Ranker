
import numpy as np
import pickle
from groq import Groq

# ✅ Load the trained K-Means model & Scaler
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ✅ Function to predict risk level
def predict_risk(patient_data):
    """
    Predicts the diabetes risk level using K-Means clustering.
    """
    # Extract feature values
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigree', 'Age']
    
    patient_array = np.array([patient_data[feature] for feature in features]).reshape(1, -1)

    # Scale the input using saved scaler
    scaled_input = scaler.transform(patient_array)

    # Predict the cluster (risk level)
    cluster = kmeans.predict(scaled_input)[0]

    # Define risk levels based on cluster
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    risk_level = risk_mapping.get(cluster, "Unknown Risk")

    return risk_level

# Initialize Groq client
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# ✅ Function to generate AI-based recommendations using Groq API
def generate_recommendation(risk_level, patient_data):
    """
    Generates a detailed medical recommendation using the AI model.
    """
    # Generate recommendation using AI
    prompt = f"""
    You are an expert diabetes specialist. Based on the following patient data, provide detailed medical recommendations:
    
    Patient Details:
    - Pregnancies: {patient_data['Pregnancies']}
    - Glucose: {patient_data['Glucose']}
    - Blood Pressure: {patient_data['BloodPressure']}
    - Skin Thickness: {patient_data['SkinThickness']}
    - Insulin: {patient_data['Insulin']}
    - BMI: {patient_data['BMI']}
    - Diabetes Pedigree Function: {patient_data['DiabetesPedigree']}
    - Age: {patient_data['Age']}
    
    Predicted Risk Level: {risk_level}
    
    Now, provide detailed medical advice including:
    - Dietary recommendations
    - Exercise guidelines
    - Lifestyle changes
    - Medical checkups needed
    - If high risk, suggest immediate medical interventions
    
    Speak like an experienced doctor explaining the condition to a patient.
    """

    # Try multiple models in case one fails
    models_to_try = [
        "llama3-1-70b-8192",  # Try the more specific model name first
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    
    for model in models_to_try:
        try:
            print(f"Attempting to use model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue  # Try the next model
    
    # If all models fail
    return "Could not generate recommendations. Please verify your Groq API key and available models."

# ✅ Sample High-Risk Patient Data
sample_patient = {
    "Pregnancies": 5,
    "Glucose": 190,   # High glucose
    "BloodPressure": 90,  # High BP
    "SkinThickness": 40,
    "Insulin": 150,
    "BMI": 35.0,  # High BMI
    "DiabetesPedigree": 0.9,
    "Age": 50
}

# 🔹 Predict Risk Level
predicted_risk = predict_risk(sample_patient)
recommendation = generate_recommendation(predicted_risk, sample_patient)
print(f"Predicted Risk Level: {predicted_risk}")
print(f"Recommendation: {recommendation}")  
