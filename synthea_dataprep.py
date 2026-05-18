#1) Synthea needs Java:

!apt-get update
!apt-get install -y openjdk-11-jdk

#2) Verify:

!java -version

# 3) Clone Synthea GitHub Repository
!git clone https://github.com/synthetichealth/synthea.git
%cd synthea

# 4) Build Synthea
!./gradlew build -x test

# 5) Run Synthea (generate data)
!sed -i 's/exporter.csv.export = false/exporter.csv.export = true/' src/main/resources/synthea.properties
!./run_synthea -p 5000
!ls output/csv

# 6) Load CSV 
import pandas as pd
patients = pd.read_csv("output/csv/patients.csv")
patients.head()

# 7) Convert CSV → patient-level records
import pandas as pd

patients = pd.read_csv("output/csv/patients.csv")
conditions = pd.read_csv("output/csv/conditions.csv")
encounters = pd.read_csv("output/csv/encounters.csv")

# group conditions per patient
patient_conditions = conditions.groupby("PATIENT")["DESCRIPTION"].apply(list)

# 8) Build structured patient profile
def build_patient_record(pid):
    patient = patients[patients["Id"] == pid].iloc[0]
    conds = patient_conditions.get(pid, [])

    return {
        "age": 2024 - int(patient["BIRTHDATE"][:4]),
        "gender": patient["GENDER"],
        "conditions": conds[:5]  # limit for simplicity
    }
# 9) Convert to natural language
def to_text(record):
    return f"""
Patient is a {record['age']}-year-old {record['gender']}.
Known conditions: {', '.join(record['conditions'])}.
"""
#10) Create instruction dataset (IMPORTANT)
dataset = []

for pid in patients["Id"].head(100):
    rec = build_patient_record(pid)
    text = to_text(rec)

    dataset.append({
        "instruction": "Summarize the patient's medical condition",
        "input": text,
        "output": f"This patient has {', '.join(rec['conditions'])}."
    })

# 11) Save for training
import json

with open("synthea_llm_data.json", "w") as f:
    json.dump(dataset, f, indent=2)