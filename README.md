# 🏢 TalentIQ: Enterprise Employee Attrition Intelligence & Risk Mitigation Suite

> A production-grade Machine Learning web application built to predict employee turnover risk, empower HR decision-making with dynamic risk threshold tuning, and provide executive-level analytics.

---

## 📌 Executive Summary
Employee turnover is a major cost driver for modern organizations. **TalentIQ** leverages historical HR metrics from the IBM HR Analytics dataset to predict individual and batch employee attrition risk. Built with an **Executive Midnight & Platinum UI/UX**, this platform bridges the gap between advanced machine learning and operational HR tooling—offering configurable risk cutoffs, automated bulk processing, and interpretability insights.

---

## 🚀 Key Features

* **👤 Single Employee Analysis Dashboard:**
  * Comprehensive interactive form categorized into *Personal & Work*, *Job & Satisfaction*, and *Career & Compensation*.
  * Real-time probability scoring with dynamic risk categorization (**Low**, **Medium**, and **High Risk**).
  * Actionable HR insight cards highlighting primary risk drivers (OverTime, Job Satisfaction, Monthly Income).
* **📁 Batch CSV Prediction Engine:**
  * Bulk upload interface supporting CSV files matching the IBM HR dataset schema.
  * Automated batch processing with summary metrics (Total Analyzed, High/Medium Risk count, Retention targets).
  * One-click CSV export for downstream reporting and HR action queues.
* **⚙️ Dynamic Decision Threshold Control:**
  * Interactive sidebar slider allowing HR stakeholders to adjust the classification threshold cutoff ($0.10$ to $0.90$) in real time to manage false positives vs. retention intervention costs.
* **🎨 Executive Midnight & Platinum UI/UX:**
  * Custom enterprise styling featuring dark-to-indigo gradients, glassmorphism cards, structured data tables, and high-contrast accessibility.

---

## 📊 Dataset & Model Architecture

* **Dataset:** IBM HR Analytics Employee Attrition & Performance Dataset ($1,470$ records, $30$ features).
* **Target Variable:** Employee Attrition (`Yes` / `No`).
* **Machine Learning Pipeline:**
  * **Algorithm:** Logistic Regression optimized via `GridSearchCV`.
  * **Validation Strategy:** Stratified 5-Fold Cross-Validation (Achieving an ROC-AUC of **81.15%**).
  * **Preprocessing:** Automated scaling and categorical encoding.

---

## 🛠️ Tech Stack

* **Core Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, Joblib
* **Web Application Framework:** Streamlit
* **Styling & UI:** Custom CSS3, Glassmorphism, Plus Jakarta Sans Typography

---

## 📂 Project Directory Structure

```text
employee-attrition-prediction/
│
├── assets/
│   └── style.css                 # Executive Midnight & Platinum stylesheet
├── models/
│   └── employee_attrition_model.joblib  # Trained production ML pipeline
├── src/
│   └── model_config.py           # Model configurations and default thresholds
├── app.py                        # Main Streamlit web application
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation