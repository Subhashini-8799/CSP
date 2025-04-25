from tkinter import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Symptom list
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 
      'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 
      'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 
      'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 
      'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
      'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps']

# Disease list
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer disease', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma',
           'Hypertension', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
           'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'Hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
           'Tuberculosis', 'Common Cold', 'Pneumonia', 'Piles', 'Heart Attack']

# Load dataset
try:
    df = pd.read_csv("Training.csv")
    tr = pd.read_csv("Testing.csv")
except FileNotFoundError:
    print("Training.csv or Testing.csv not found.")
    exit()

# Map disease names to integers and reverse
disease_mapping = {d: i for i, d in enumerate(disease)}
reverse_mapping = {i: d for d, i in disease_mapping.items()}

# Clean and encode the prognosis column
df = df[df['prognosis'].isin(disease_mapping)]
df['prognosis'] = df['prognosis'].map(disease_mapping)
df.dropna(subset=['prognosis'], inplace=True)

tr = tr[tr['prognosis'].isin(disease_mapping)]
tr['prognosis'] = tr['prognosis'].map(disease_mapping)
tr.dropna(subset=['prognosis'], inplace=True)

# Ensure only existing symptoms are used
existing_symptoms = [symptom for symptom in l1 if symptom in df.columns]

# Feature selection
X = df[existing_symptoms]
y = df["prognosis"]
X_test = tr[existing_symptoms]
y_test = tr["prognosis"]

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# GUI Setup
root = Tk()
root.configure(background='light green')
root.title("Disease Predictor")

Symptom1, Symptom2, Symptom3, Symptom4, Symptom5 = [StringVar(value="None") for _ in range(5)]
Name = StringVar()

# Heading
Label(root, text="Disease Predictor using Machine Learning", fg="white", bg="light green",
      font=("Elephant", 20)).grid(row=1, column=0, columnspan=2, padx=100, pady=20)

# Labels
Label(root, text="Name of the Patient", fg="yellow", bg="black").grid(row=6, column=0, pady=15, sticky=W)
Label(root, text="Symptom 1", fg="yellow", bg="black").grid(row=7, column=0, pady=10, sticky=W)
Label(root, text="Symptom 2", fg="yellow", bg="black").grid(row=8, column=0, pady=10, sticky=W)
Label(root, text="Symptom 3", fg="yellow", bg="black").grid(row=9, column=0, pady=10, sticky=W)
Label(root, text="Symptom 4", fg="yellow", bg="black").grid(row=10, column=0, pady=10, sticky=W)
Label(root, text="Symptom 5", fg="yellow", bg="black").grid(row=11, column=0, pady=10, sticky=W)

# Dropdowns
OPTIONS = sorted(existing_symptoms)
Entry(root, textvariable=Name).grid(row=6, column=1)
OptionMenu(root, Symptom1, *OPTIONS).grid(row=7, column=1)
OptionMenu(root, Symptom2, *OPTIONS).grid(row=8, column=1)
OptionMenu(root, Symptom3, *OPTIONS).grid(row=9, column=1)
OptionMenu(root, Symptom4, *OPTIONS).grid(row=10, column=1)
OptionMenu(root, Symptom5, *OPTIONS).grid(row=11, column=1)

# Predict Function
def randomforest():
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    if all(s == "None" for s in psymptoms):
        t1.delete("1.0", END)
        t1.insert(END, "Please select at least one symptom.")
        return
    
    l2 = [1 if symptom in psymptoms else 0 for symptom in existing_symptoms]
    inputtest = [l2]
    prediction = clf.predict(inputtest)[0]
    predicted_disease = reverse_mapping.get(prediction, "Unknown")
    
    t1.delete("1.0", END)
    t1.insert(END, predicted_disease)

# Predict Button
Button(root, text="Predict Disease", command=randomforest, bg="green", fg="yellow").grid(row=13, column=1, pady=20)

# Output Textbox
t1 = Text(root, height=1, width=40, bg="orange", fg="black")
t1.grid(row=15, column=1, padx=10, pady=10)

root.mainloop()
