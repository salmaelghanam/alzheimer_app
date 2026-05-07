# 🧠 Alzheimer's Detection System

A multi-step AI-powered web application for early Alzheimer's detection, combining clinical data analysis and MRI brain scan classification using Deep Learning and Machine Learning models.

> ⚠️ For research and educational purposes only. Not a substitute for professional medical diagnosis.

---

## ✨ How It Works

The system follows a 3-step diagnostic workflow:

**Step 1 — Clinical Data Analysis**
The user enters demographic, lifestyle, and cognitive assessment data. Multiple ML/DL models analyze the input and produce a probability score.

**Step 2 — MRI Scan (if needed)**
If clinical confidence for "No Alzheimer's" is below 80%, the system requests an MRI brain scan upload for further analysis using deep learning image classifiers.

**Step 3 — Final Diagnosis**
Results from both steps are fused (Clinical 40% + MRI 60%) to produce a final diagnosis with a confidence score.

---

## 🤖 Models Used

### Clinical Models (tabular data)
- Random Forest
- XGBoost
- 1D-CNN

### MRI Models (image classification — 4 classes)
- VGG19
- ResNet50
- EfficientNetB0

**MRI Classes:** `NonDemented` · `VeryMildDemented` · `MildDemented` · `ModerateDemented`

> **Note:** Trained model files (`.keras`, `.pkl`) are not included in this repo due to file size limits. Place them in a `saved_models/` folder to run the app locally.

---

## 🖥️ Tech Stack

- **Frontend:** Streamlit
- **Deep Learning:** TensorFlow / Keras
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** NumPy, Pandas, Pillow, Joblib

---

## 📂 Project Structure

```
alzheimer_app/
│
├── app.py                      # Main Streamlit application
├── alzahaimer nootbook .ipynb  # Model training notebook
├── requirements.txt            # Python dependencies
├── .gitignore
└── saved_models/               # (not included — add locally)
    ├── VGG19_best.keras
    ├── ResNet50_best.keras
    ├── EfficientNetB0_best.keras
    ├── 1DCNN_clinical_best.keras
    ├── RandomForest_clinical.pkl
    ├── XGBoost_clinical.pkl
    └── clinical_scaler.pkl
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add trained models

Place your trained model files inside a `saved_models/` folder in the project root.

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📋 Clinical Input Features

The app collects the following inputs:

**Demographics:** Age, Gender, Education Level

**Lifestyle:** BMI, Smoking, Alcohol consumption, Physical activity

**Medical History:** Family history of Alzheimer's, Diabetes, Hypertension, Depression

**Cognitive Assessments:** MMSE score, MoCA score, ADL score, Functional Assessment

**Clinical Measurements:** Total Cholesterol, Systolic BP, Diastolic BP

---

## 👩‍💻 Author

**Salma Elghanam**
Computer Science & AI Student
[GitHub Profile](https://github.com/salmaelghanam)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
