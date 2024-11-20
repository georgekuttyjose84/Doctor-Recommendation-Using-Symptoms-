import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.naive_bayes import GaussianNB

# Load and preprocess the data
data = read_csv("Datasets/Training.csv")

x = data.drop('prognosis', axis=1)
y = data['prognosis']

# Train the Naive Bayes model
gnb = GaussianNB()
gnb.fit(x, y)

# List of symptoms
list_a = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain',
    'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever',
    'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
    'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
    'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum',
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
    'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# Disease-to-department mapping
disease_department_mapping = {
    "Fungal infection": "Dermatology",
    "Allergy": "Immunology/Allergy Specialist",
    "GERD": "Gastroenterology",
    "Chronic cholestasis": "Hepatology/Gastroenterology",
    "Drug Reaction": "Dermatology/Immunology",
    "Peptic ulcer disease": "Gastroenterology",
    "AIDS": "Infectious Diseases",
    "Diabetes": "Endocrinology",
    "Gastroenteritis": "Gastroenterology",
    "Bronchial Asthma": "Pulmonology/Immunology",
    "Hypertension": "Cardiology",
    "Migraine": "Neurology",
    "Cervical spondylosis": "Orthopedics/Neurology",
    "Paralysis (brain hemorrhage)": "Neurology/Neurosurgery",
    "Jaundice": "Hepatology/Gastroenterology",
    "Malaria": "Infectious Diseases",
    "Chicken pox": "Infectious Diseases/Dermatology",
    "Dengue": "Infectious Diseases",
    "Typhoid": "Infectious Diseases",
    "Hepatitis A": "Hepatology",
    "Hepatitis B": "Hepatology",
    "Hepatitis C": "Hepatology",
    "Hepatitis D": "Hepatology",
    "Hepatitis E": "Hepatology",
    "Alcoholic hepatitis": "Hepatology",
    "Tuberculosis": "Pulmonology/Infectious Diseases",
    "Common Cold": "General Medicine",
    "Pneumonia": "Pulmonology",
    "Dimorphic hemorrhoids (piles)": "Proctology/Surgery",
    "Heart attack": "Cardiology",
    "Varicose veins": "Vascular Surgery",
    "Hypothyroidism": "Endocrinology",
    "Hyperthyroidism": "Endocrinology",
    "Hypoglycemia": "Endocrinology",
    "Osteoarthritis": "Orthopedics",
    "Arthritis": "Rheumatology",
    "(Vertigo) Paroxysmal Positional Vertigo": "Neurology/Ear, Nose, Throat (ENT)",
    "Acne": "Dermatology",
    "Urinary tract infection": "Urology",
    "Psoriasis": "Dermatology",
    "Impetigo": "Dermatology"
}

# Function to predict the disease based on selected symptoms
def predict_disease(selected_symptoms):
    list_c = [0] * len(list_a)
    for symptom in selected_symptoms:
        if symptom in list_a:
            list_c[list_a.index(symptom)] = 1
    test = np.array(list_c).reshape(1, -1)
    prediction = gnb.predict(test)[0]
    department = disease_department_mapping.get(prediction, "General Medicine")
    disease_label.config(text=prediction)
    doctor_label.config(text=department)

# Function to get all symptoms and make predictions
def check_symptoms():
    selected_symptoms = [var.get() for var in symptom_vars if var.get() != "Select"]
    if len(selected_symptoms) < 2:
        messagebox.showwarning("Warning", "Please select at least 2 symptoms.")
        return
    if len(selected_symptoms) != len(set(selected_symptoms)):
        messagebox.showwarning("Warning", "Duplicate symptoms are not allowed!")
        return
    predict_disease(selected_symptoms)

# Function to add more symptom dropdowns
def add_symptom():
    new_var = tk.StringVar(value="Select")
    symptom_vars.append(new_var)
    row = len(symptom_vars) - 1  # Calculate row dynamically
    label = tk.Label(symptom_frame, text=f"Symptom {len(symptom_vars)}", font=label_font)
    label.grid(row=row, column=0, pady=5, sticky="W")
    symptom_labels.append(label)  # Track labels for resetting
    dropdown = ttk.OptionMenu(symptom_frame, new_var, "Select", *list_a)
    dropdown.grid(row=row, column=1, pady=5, sticky="W")
    symptom_dropdowns.append(dropdown)  # Track dropdowns for resetting
    symptom_frame.update_idletasks()

# Function to reset all fields
def reset_fields():
    # Reset default symptom variables
    for var in symptom_vars[:5]:
        var.set("Select")
    
    # Remove dynamically added labels and dropdowns
    for label in symptom_labels[5:]:
        label.destroy()
    for dropdown in symptom_dropdowns[5:]:
        dropdown.destroy()
    
    # Keep only default variables and elements
    del symptom_vars[5:]
    del symptom_labels[5:]
    del symptom_dropdowns[5:]
    
    disease_label.config(text="")
    doctor_label.config(text="")

# Main tkinter application
root = tk.Tk()
root.title("Disease Prediction From Symptoms")
root.geometry("600x600")
root.eval('tk::PlaceWindow . center')

# Font configuration for better visibility
label_font = ("Arial", 14, "bold")
dropdown_font = ("Arial", 12)

# Heading
heading = tk.Label(root, text="Doctor Recommendation Using Symptoms", font=("Arial", 18, "bold"))
heading.pack(pady=10)

# Frame for symptoms
symptom_frame = tk.Frame(root)
symptom_frame.pack(pady=10)

# Initialize symptom variables
symptom_vars = [tk.StringVar(value="Select") for _ in range(5)]
symptom_labels = []
symptom_dropdowns = []

# Create default dropdowns for symptoms
for i, var in enumerate(symptom_vars):
    label = tk.Label(symptom_frame, text=f"Symptom {i+1}", font=label_font)
    label.grid(row=i, column=0, pady=5, sticky="W")
    symptom_labels.append(label)
    dropdown = ttk.OptionMenu(symptom_frame, var, "Select", *list_a)
    dropdown.grid(row=i, column=1, pady=5, sticky="W")
    symptom_dropdowns.append(dropdown)

# Add symptom button
add_symptom_btn = tk.Button(root, text="Add Another Symptom", command=add_symptom, font=label_font)
add_symptom_btn.pack(pady=10)

# Predict and Reset buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Predict", command=check_symptoms, font=label_font).grid(row=0, column=0, padx=20)
tk.Button(button_frame, text="Reset", command=reset_fields, font=label_font).grid(row=0, column=1, padx=20)

# Result labels
result_frame = tk.Frame(root)
result_frame.pack(pady=20)

tk.Label(result_frame, text="DISEASE:", font=label_font).grid(row=0, column=0, pady=10, sticky="E")
tk.Label(result_frame, text="DOCTOR:", font=label_font).grid(row=1, column=0, pady=10, sticky="E")

disease_label = tk.Label(result_frame, text="", font=dropdown_font)
disease_label.grid(row=0, column=1, pady=10)

doctor_label = tk.Label(result_frame, text="", font=dropdown_font)
doctor_label.grid(row=1, column=1, pady=10)

root.mainloop()
