import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate patient data
patient_ids = [f"PT{str(i).zfill(6)}" for i in range(1, 101)]
first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth", 
               "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen",
               "Emma", "Noah", "Olivia", "Liam", "Ava", "Sophia", "Jackson", "Isabella", "Aiden", "Mia"]
last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
              "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
              "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King"]
genders = ["Male", "Female"]
blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
conditions = ["Hypertension", "Diabetes Type 2", "Asthma", "Arthritis", "Anxiety", "Depression", 
              "Obesity", "COPD", "Coronary Artery Disease", "Chronic Kidney Disease", None]

# Function to generate random date within a range
def random_date(start_date, end_date):
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randrange(days_between)
    return start_date + timedelta(days=random_days)

# Generate patients dataframe
patients_data = []
for patient_id in patient_ids:
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    gender = random.choice(genders)
    age = random.randint(18, 90)
    blood_type = random.choice(blood_types)
    primary_condition = random.choice(conditions)
    registration_date = random_date(datetime(2018, 1, 1), datetime(2023, 12, 31)).strftime('%Y-%m-%d')
    
    patients_data.append({
        "patient_id": patient_id,
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}",
        "gender": gender,
        "age": age,
        "blood_type": blood_type,
        "primary_condition": primary_condition,
        "registration_date": registration_date
    })

patients_df = pd.DataFrame(patients_data)
patients_df.to_csv("test_data/csv/patients.csv", index=False)
print("Generated patients.csv with", len(patients_df), "records")

# Generate medication data
medication_ids = [f"MED{str(i).zfill(5)}" for i in range(1, 51)]
medication_names = [
    "Lisinopril", "Metformin", "Atorvastatin", "Levothyroxine", "Amlodipine", "Metoprolol", "Omeprazole", 
    "Simvastatin", "Losartan", "Gabapentin", "Hydrochlorothiazide", "Sertraline", "Montelukast", "Pantoprazole", 
    "Escitalopram", "Acetaminophen", "Ibuprofen", "Aspirin", "Fluoxetine", "Fluticasone", "Albuterol", 
    "Prednisone", "Tramadol", "Furosemide", "Azithromycin", "Rosuvastatin", "Carvedilol", "Duloxetine", 
    "Citalopram", "Warfarin", "Clopidogrel", "Clonazepam", "Meloxicam", "Hydrocodone", "Venlafaxine", 
    "Amoxicillin", "Alprazolam", "Bupropion", "Ciprofloxacin", "Trazodone", "Oxycodone", "Lorazepam", 
    "Cyclobenzaprine", "Pravastatin", "Methylprednisolone", "Diazepam", "Lyrica", "Xyzal", "Vyvanse", "Januvia"
]
categories = ["Antihypertensive", "Antidiabetic", "Statin", "Thyroid Medication", "Calcium Channel Blocker", 
              "Beta Blocker", "Proton Pump Inhibitor", "Statin", "Antihypertensive", "Anticonvulsant", 
              "Diuretic", "SSRI", "Asthma Medication", "Proton Pump Inhibitor", "SSRI", "Analgesic", 
              "NSAID", "Antiplatelet", "SSRI", "Corticosteroid", "Bronchodilator", "Corticosteroid"]
manufacturers = ["Pfizer", "Merck", "Novartis", "Roche", "Johnson & Johnson", "AbbVie", "Sanofi", 
                "GlaxoSmithKline", "AstraZeneca", "Gilead Sciences", "Amgen", "Bristol-Myers Squibb"]

# Generate medications dataframe
medications_data = []
for i, medication_id in enumerate(medication_ids):
    medication_name = medication_names[i % len(medication_names)]
    category = categories[i % len(categories)]
    manufacturer = random.choice(manufacturers)
    approval_year = random.randint(1980, 2022)
    
    medications_data.append({
        "medication_id": medication_id,
        "medication_name": medication_name,
        "category": category,
        "manufacturer": manufacturer,
        "approval_year": approval_year
    })

medications_df = pd.DataFrame(medications_data)
medications_df.to_csv("test_data/csv/medications.csv", index=False)
print("Generated medications.csv with", len(medications_df), "records")


# Generate prescription data
prescriptions_data = []
for _ in range(300):  # Create 300 prescriptions
    patient_idx = random.randint(0, len(patients_data) - 1)
    patient_id = patients_data[patient_idx]["patient_id"]
    patient_name = patients_data[patient_idx]["full_name"]
    
    medication_idx = random.randint(0, len(medications_data) - 1)
    medication_id = medications_data[medication_idx]["medication_id"]
    medication_name = medications_data[medication_idx]["medication_name"]
    
    dosage = f"{random.choice([5, 10, 20, 25, 50, 100, 150, 200, 250, 500])} mg"
    frequency = random.choice(["Once daily", "Twice daily", "Three times daily", "As needed", "Weekly", "Every 12 hours"])
    
    prescription_date = random_date(datetime(2020, 1, 1), datetime(2023, 12, 31)).strftime('%Y-%m-%d')
    
    prescriptions_data.append({
        "prescription_id": f"RX{str(len(prescriptions_data) + 1).zfill(6)}",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "medication_id": medication_id,
        "medication_name": medication_name,
        "dosage": dosage,
        "frequency": frequency,
        "prescription_date": prescription_date,
        "prescribing_doctor": random.choice(["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Elizabeth Taylor", 
                                             "Dr. David Rodriguez", "Dr. Emily Wilson", "Dr. Robert Brown"])
    })

prescriptions_df = pd.DataFrame(prescriptions_data)
prescriptions_df.to_csv("test_data/csv/prescriptions.csv", index=False)
print("Generated prescriptions.csv with", len(prescriptions_df), "records")


# Generate doctor data
doctor_ids = [f"DOC{str(i).zfill(4)}" for i in range(1, 31)]
doctor_titles = ["Dr."]
doctor_first_names = ["Sarah", "Michael", "Elizabeth", "David", "Emily", "Robert", "Jennifer", "William", 
                      "Amanda", "Thomas", "Sophia", "James", "Olivia", "Richard", "Emma", "John", "Anna", 
                      "Steven", "Maria", "Daniel", "Jessica", "Charles", "Margaret", "Christopher", "Lisa"]
doctor_last_names = ["Johnson", "Chen", "Taylor", "Rodriguez", "Wilson", "Brown", "Miller", "Davis", 
                     "Garcia", "Martinez", "Smith", "Anderson", "Jackson", "Thompson", "White", "Harris", 
                     "Martin", "Jones", "Williams", "Lee", "Walker", "Hall", "Allen", "Young", "King"]
specialties = ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Orthopedics", "Internal Medicine", 
               "Dermatology", "Endocrinology", "Gastroenterology", "Psychiatry", "Radiology", "Nephrology"]
hospitals = ["Memorial Hospital", "University Medical Center", "General Hospital", "St. Mary's", 
             "Metropolitan Medical Center", "Community Health Center", "City Hospital", "Regional Medical Center"]

# Generate doctors dataframe
doctors_data = []
for i, doctor_id in enumerate(doctor_ids):
    title = "Dr."
    first_name = doctor_first_names[i % len(doctor_first_names)]
    last_name = doctor_last_names[i % len(doctor_last_names)]
    specialty = random.choice(specialties)
    hospital = random.choice(hospitals)
    years_of_experience = random.randint(1, 35)
    
    doctors_data.append({
        "doctor_id": doctor_id,
        "title": title,
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{title} {first_name} {last_name}",
        "specialty": specialty,
        "hospital": hospital,
        "years_of_experience": years_of_experience
    })

doctors_df = pd.DataFrame(doctors_data)
doctors_df.to_csv("test_data/csv/doctors.csv", index=False)
print("Generated doctors.csv with", len(doctors_df), "records")
def generate_clinical_notes():
    notes = []
    
    # Get all medication names for reference
    all_medication_names = [med["medication_name"] for med in medications_data]
    
    # Use a subset of patients and medications from our structured data
    selected_patients = random.sample(patients_data, 15)
    
    for i, patient in enumerate(selected_patients):
        patient_name = patient["full_name"]
        patient_id = patient["patient_id"]
        gender = patient["gender"]
        pronoun = "She" if gender == "Female" else "He"
        pronoun_obj = "her" if gender == "Female" else "his"
        age = patient["age"]
        condition = patient["primary_condition"] or random.choice(conditions[:-1])  # Exclude None
        
        # Choose random medications for this patient
        patient_medications = random.sample(medications_data, random.randint(1, 3))
        medication_names = [med["medication_name"] for med in patient_medications]
        
        # Create a list of medications not prescribed to this patient
        other_medications = [m for m in all_medication_names if m not in medication_names]
        
        # Choose a random doctor
        doctor = random.choice(doctors_data)
        doctor_name = doctor["full_name"]
        
        # Generate a clinical note
        date = random_date(datetime(2021, 1, 1), datetime(2023, 12, 31)).strftime('%B %d, %Y')
        
        note = f"""
CLINICAL NOTE
Date: {date}
Patient: {patient_name} (ID: {patient_id})
Attending Physician: {doctor_name}

SUBJECTIVE:
{patient_name} is a {age}-year-old {gender.lower()} presenting with {condition.lower()}. {pronoun} reports {random.choice(["feeling tired throughout the day", "occasional dizziness", "persistent headaches", "joint pain", "difficulty sleeping", "shortness of breath", "chest discomfort", "nausea after eating"])}. {pronoun} has been experiencing these symptoms for {random.choice(["several days", "about a week", "the past month", "several months"])}.

OBJECTIVE:
Vital signs: 
- BP: {random.randint(110, 140)}/{random.randint(70, 90)} mmHg
- HR: {random.randint(60, 100)} bpm
- Temp: {round(random.uniform(36.5, 37.5), 1)}°C
- RR: {random.randint(12, 20)} breaths/min

Physical examination reveals {random.choice(["no acute distress", "mild discomfort", "normal findings", "slight tenderness in the abdomen", "clear lungs", "regular heart rhythm with no murmurs"])}. {random.choice(["Laboratory tests were ordered.", "No additional tests were necessary at this time.", "Follow-up labs will be scheduled."])}

ASSESSMENT & PLAN:
Patient diagnosed with {condition}. {pronoun} is currently taking {', '.join(medication_names[:-1]) + ' and ' + medication_names[-1] if len(medication_names) > 1 else medication_names[0]}. {random.choice([
    f"Will continue current medication regimen.", 
    f"Adjusting {pronoun_obj} dosage of {random.choice(medication_names)}.",
    f"Adding {random.choice(other_medications) if other_medications else 'a new medication'} to {pronoun_obj} current medications."
])}

{random.choice(["Patient will follow up in 3 months.", "Scheduled for follow-up in 2 weeks.", "Will reassess in 1 month.", "Patient advised to return if symptoms worsen."])}

Signed: {doctor_name}
"""
        notes.append(note)
    
    return "\n\n" + "="*80 + "\n\n".join(notes)


def generate_research_paper():
    # Choose some medications to focus on
    focus_medications = random.sample(medications_data, 5)
    medication_names = [med["medication_name"] for med in focus_medications]
    
    # Generate a research paper about these medications
    title = f"Comparative Effectiveness of {medication_names[0]} and {medication_names[1]} in the Treatment of {random.choice(conditions[:-1])}"
    
    authors = []
    for _ in range(3):
        doctor = random.choice(doctors_data)
        authors.append(f"{doctor['first_name'][0]}. {doctor['last_name']}")
    
    paper = f"""
{title}

{', '.join(authors)}
{random.choice(["University Medical Center", "Research Institute of Medicine", "National Health Research Laboratory"])}

ABSTRACT
This study compares the effectiveness of {medication_names[0]} and {medication_names[1]} in the treatment of patients with {random.choice(conditions[:-1])}. A total of {random.randint(100, 500)} patients were enrolled in this double-blind, randomized controlled trial over a period of {random.randint(12, 36)} months. Results indicate that {random.choice(medication_names[:2])} demonstrated {random.choice(["superior", "comparable", "slightly better"])} efficacy compared to {random.choice([m for m in medication_names[:2] if m != random.choice(medication_names[:2])])} in terms of {random.choice(["symptom reduction", "side effect profile", "patient tolerance", "long-term outcomes"])}.

INTRODUCTION
The management of {random.choice(conditions[:-1])} continues to present challenges in clinical practice. Current treatment options include {', '.join(medication_names[:-1])} and {medication_names[-1]}. This study aims to provide a comprehensive comparison of {medication_names[0]} and {medication_names[1]}, which are commonly prescribed but lack head-to-head comparison data.

METHODS
We conducted a prospective, randomized, double-blind trial involving {random.randint(100, 500)} patients diagnosed with {random.choice(conditions[:-1])}. Patients were randomly assigned to receive either {medication_names[0]} ({random.choice(["5", "10", "20", "50", "100"])} mg daily) or {medication_names[1]} ({random.choice(["5", "10", "20", "50", "100"])} mg daily) for {random.randint(12, 24)} weeks. Primary outcomes included {random.choice(["reduction in symptoms", "improvement in quality of life", "time to remission", "adverse event rates"])}.

RESULTS
Of the {random.randint(100, 500)} patients enrolled, {random.randint(80, 95)}% completed the study. Patients receiving {random.choice(medication_names[:2])} showed a {random.randint(10, 40)}% improvement in primary outcomes compared to {random.randint(5, 35)}% in the {random.choice([m for m in medication_names[:2] if m != random.choice(medication_names[:2])])} group (p < {random.choice(["0.001", "0.01", "0.05"])}).

Adverse events were reported in {random.randint(5, 25)}% of patients in the {medication_names[0]} group and {random.randint(5, 25)}% in the {medication_names[1]} group. The most common side effects included {random.choice(["headache", "nausea", "dizziness", "fatigue"])} and {random.choice(["insomnia", "gastrointestinal discomfort", "rash", "dry mouth"])}.

DISCUSSION
Our findings suggest that {random.choice(medication_names[:2])} may offer advantages over {random.choice([m for m in medication_names[:2] if m != random.choice(medication_names[:2])])} in the treatment of {random.choice(conditions[:-1])}, particularly in {random.choice(["elderly patients", "patients with comorbidities", "treatment-resistant cases", "newly diagnosed patients"])}.

It is worth noting that {medication_names[2]} and {medication_names[3]}, which were not directly evaluated in this study, have shown promise in previous research. Future studies comparing all available treatment options would be valuable.

CONCLUSION
{random.choice(medication_names[:2])} demonstrates {random.choice(["superior", "promising", "effective"])} results in the management of {random.choice(conditions[:-1])}. Clinicians should consider {random.choice(["patient-specific factors", "comorbidities", "side effect profiles", "cost considerations"])} when choosing between {medication_names[0]} and {medication_names[1]}.

REFERENCES
1. Smith J, et al. (2021). Treatment guidelines for {random.choice(conditions[:-1])}. Journal of Medicine, 45(3), 234-241.
2. Johnson A, Brown T. (2022). Efficacy of {medication_names[2]} in chronic disease management. Clinical Therapeutics, 28(2), 112-120.
3. Williams R, et al. (2020). Comparative analysis of {medication_names[0]} and {medication_names[3]}. New England Journal of Medicine, 382(12), 1089-1098.
"""
    return paper

research_paper = generate_research_paper()

with open("test_data/pdf/research_paper.txt", "w") as f:
    f.write(research_paper)

print("Generated research_paper.txt")


def generate_drug_interactions():
    # Select a subset of medications
    selected_medications = random.sample(medications_data, 15)
    
    document = """
MEDICATION INTERACTION GUIDE
Hospital Pharmacy Department
Last Updated: June 15, 2023

This document provides information about potential interactions between commonly prescribed medications. 
It is intended as a reference for healthcare professionals and should not replace clinical judgment.

KNOWN INTERACTIONS:
"""
    
    # Generate some random interactions
    for i in range(20):
        med1 = random.choice(selected_medications)
        # Ensure med2 is different from med1
        med2 = random.choice([m for m in selected_medications if m != med1])
        
        interaction_severity = random.choice(["Minor", "Moderate", "Severe"])
        
        interaction = f"""
{med1['medication_name']} + {med2['medication_name']}
Severity: {interaction_severity}
Effect: {random.choice([
    "May increase risk of bleeding", 
    "May decrease effectiveness", 
    "May enhance hypotensive effects", 
    "May increase risk of serotonin syndrome", 
    "May elevate serum potassium levels",
    "May prolong QT interval",
    "May increase risk of myopathy",
    "May reduce anticoagulant effectiveness",
    "May cause CNS depression",
    "May reduce absorption"
])}
Recommendation: {random.choice([
    "Monitor closely", 
    "Avoid combination if possible", 
    "Adjust dosage as needed", 
    "Consider alternative therapy",
    "Monitor blood levels regularly",
    "Monitor blood pressure",
    "Monitor for signs of toxicity",
    "Space dosing by at least 2 hours",
    "Monitor renal function"
])}
"""
        document += interaction
    
    document += """
SPECIAL POPULATIONS:

Pregnant Women:
The following medications should be used with caution or avoided during pregnancy:
"""
    
    # Add some medications with pregnancy warnings
    pregnancy_meds = random.sample(selected_medications, 5)
    for med in pregnancy_meds:
        document += f"- {med['medication_name']}: {random.choice(['Contraindicated', 'Use only if benefits outweigh risks', 'Limited data available', 'Generally considered safe'])}\n"
    
    document += """
Elderly Patients:
The following medications require dose adjustments or special monitoring in elderly patients:
"""
    
    # Add some medications with elderly patient considerations
    elderly_meds = random.sample(selected_medications, 5)
    for med in elderly_meds:
        document += f"- {med['medication_name']}: {random.choice(['Start at lower dose', 'Monitor renal function', 'Increased risk of falls', 'May cause confusion', 'Monitor hepatic function'])}\n"
    
    return document

drug_interactions = generate_drug_interactions()

with open("test_data/pdf/drug_interactions.txt", "w") as f:
    f.write(drug_interactions)

print("Generated drug_interactions.txt")


import json

schema = {
    "entity_types": [
        {
            "name": "Patient",
            "description": "A person receiving medical care",
            "properties": ["patient_id", "first_name", "last_name", "full_name", "gender", "age", "blood_type", "registration_date"]
        },
        {
            "name": "Medication",
            "description": "A pharmaceutical drug or treatment",
            "properties": ["medication_id", "medication_name", "category", "manufacturer", "approval_year"]
        },
        {
            "name": "Doctor",
            "description": "A medical professional",
            "properties": ["doctor_id", "full_name", "specialty", "hospital", "years_of_experience"]
        },
        {
            "name": "Condition",
            "description": "A medical condition or disease",
            "properties": ["name", "description", "icd_code", "category"]
        },
        {
            "name": "Prescription",
            "description": "A prescription for medication",
            "properties": ["prescription_id", "dosage", "frequency", "prescription_date"]
        }
    ],
    "relation_types": [
        {
            "name": "HAS_CONDITION",
            "description": "Patient has a medical condition",
            "source_types": ["Patient"],
            "target_types": ["Condition"]
        },
        {
            "name": "PRESCRIBED",
            "description": "Doctor prescribed a medication to a patient",
            "source_types": ["Doctor"],
            "target_types": ["Prescription"]
        },
        {
            "name": "PRESCRIBED_FOR",
            "description": "Prescription is for a patient",
            "source_types": ["Prescription"],
            "target_types": ["Patient"]
        },
        {
            "name": "CONTAINS",
            "description": "Prescription contains a medication",
            "source_types": ["Prescription"],
            "target_types": ["Medication"]
        },
        {
            "name": "TREATS",
            "description": "Medication treats a condition",
            "source_types": ["Medication"],
            "target_types": ["Condition"]
        },
        {
            "name": "INTERACTS_WITH",
            "description": "Medication interacts with another medication",
            "source_types": ["Medication"],
            "target_types": ["Medication"]
        }
    ]
}

with open("test_data/medical_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("Generated medical_schema.json")




# Example command line usage
python -m unified_kg.main --csv test_data/csv/patients.csv test_data/csv/medications.csv test_data/csv/prescriptions.csv test_data/csv/doctors.csv --pdf  test_data/pdf/research_paper.pdf test_data/pdf/drug_interactions.pdf --schema test_data/medical_schema.json --output results.json