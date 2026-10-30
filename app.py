import os
import joblib
import pandas as pd
import streamlit as st

# Try importing default threshold, fallback to 0.40
try:
    from src.model_config import DECISION_THRESHOLD as INITIAL_THRESHOLD
except (ImportError, ModuleNotFoundError):
    INITIAL_THRESHOLD = 0.40


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD CUSTOM CSS
# =========================================================

def load_css():
    css_path = os.path.join("assets", "style.css")
    if not os.path.exists(css_path):
        css_path = "style.css"  # fallback if in root

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/employee_attrition_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(
        f"Unable to load the trained model.\n\nError: {e}"
    )
    st.stop()


# =========================================================
# SIDEBAR & DYNAMIC DECISION THRESHOLD SLIDER
# =========================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-title">
        📌 Model Information
    </div>
    """)

    st.html("""
    <div class="sidebar-info">
        <p><strong>Algorithm:</strong> Logistic Regression</p>
        <p><strong>Optimization:</strong> GridSearchCV</p>
        <p><strong>Cross Validation:</strong> Stratified 5-Fold</p>
        <p><strong>ROC-AUC:</strong> 81.15%</p>
    </div>
    """)

    st.markdown("---")

    st.html("""
    <div class="sidebar-title">
        ⚙️ Decision Threshold Control
    </div>
    """)
    
    # Interactive Slider for Decision Threshold
    DECISION_THRESHOLD = st.slider(
        "Adjust Risk Cutoff",
        min_value=0.10,
        max_value=0.90,
        value=float(INITIAL_THRESHOLD),
        step=0.05,
        help="Changes the sensitivity threshold for classifying medium/high attrition risk."
    )

    st.markdown("---")

    st.html("""
    <div class="sidebar-title">
        📊 Dataset
    </div>
    """)

    st.html("""
    <div class="sidebar-info">
        <p><strong>Records:</strong> 1,470</p>
        <p><strong>Features:</strong> 30</p>
        <p><strong>Target:</strong> Employee Attrition</p>
    </div>
    """)

    st.markdown("---")

    st.html("""
    <div class="sidebar-footer">
        TalentIQ Analytics Suite<br>
        Powered by Machine Learning
    </div>
    """)


# =========================================================
# MAIN HEADER
# =========================================================

st.html("""
<div class="main-header">
    <h1>📊 Employee Attrition Prediction</h1>
    <p>
        Predict whether an employee is likely to leave the
        organization using a Machine Learning model trained
        on the IBM HR Analytics dataset.
    </p>
</div>
""")


# =========================================================
# TABS FOR NAVIGATION (SINGLE VS BATCH)
# =========================================================

tab1, tab2 = st.tabs(["👤 Single Employee Analysis", "📁 Batch CSV Prediction"])


# =========================================================
# TAB 1: SINGLE EMPLOYEE ANALYSIS
# =========================================================

with tab1:

    st.html("""
    <div class="section-heading">
        👤 Employee Information
    </div>
    <div class="section-description">
        Enter the employee's information below to estimate attrition risk.
    </div>
    """)

    # THREE COLUMN LAYOUT
    col1, col2, col3 = st.columns(3, gap="large")

    # COLUMN 1: PERSONAL & WORK
    with col1:
        st.html("""<div class="column-title">Personal & Work</div>""")

        age = st.number_input("Age", min_value=18, max_value=60, value=31, step=1, key="s_age")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0, key="s_gender")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=0, key="s_marital")
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], index=1, key="s_travel")
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"], index=0, key="s_dept")
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ], index=0, key="s_role")
        distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=29, value=20, step=1, key="s_dist")

    # COLUMN 2: JOB & SATISFACTION
    with col2:
        st.html("""<div class="column-title">Job & Satisfaction</div>""")

        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=0, key="s_jlevel")
        job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=0, key="s_jinvolve")
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=0, key="s_jsat")
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=0, key="s_esat")
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=0, key="s_rsat")
        work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4], index=0, key="s_wlb")
        overtime = st.selectbox("Overtime", ["Yes", "No"], index=0, key="s_ot")
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=0, key="s_sol")

    # COLUMN 3: CAREER & COMPENSATION
    with col3:
        st.html("""<div class="column-title">Career & Compensation</div>""")

        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=2500, step=100, key="s_mincome")
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=15000, step=100, key="s_mrate")
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800, step=10, key="s_drate")
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65, step=1, key="s_hrate")
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=6, step=1, key="s_twy")
        years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=1, step=1, key="s_yac")
        years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=18, value=0, step=1, key="s_yicr")
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, value=0, step=1, key="s_ywcm")

    st.markdown("---")
    st.html("""<div class="section-heading">📚 Additional Information</div>""")

    additional_col1, additional_col2, additional_col3 = st.columns(3, gap="large")

    with additional_col1:
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=0, key="s_edu")
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], index=0, key="s_edufield")
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=5, step=1, key="s_ncw")

    with additional_col2:
        performance_rating = st.selectbox("Performance Rating", [3, 4], index=0, key="s_pr")
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=11, step=1, key="s_psh")
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2, step=1, key="s_ttly")

    with additional_col3:
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, step=1, key="s_yslp")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Predict Attrition Risk", use_container_width=True, key="single_predict")

    if predict_button:
        employee_data = pd.DataFrame({
            "Age": [age], "BusinessTravel": [business_travel], "DailyRate": [daily_rate],
            "Department": [department], "DistanceFromHome": [distance_from_home], "Education": [education],
            "EducationField": [education_field], "EnvironmentSatisfaction": [environment_satisfaction],
            "Gender": [gender], "HourlyRate": [hourly_rate], "JobInvolvement": [job_involvement],
            "JobLevel": [job_level], "JobRole": [job_role], "JobSatisfaction": [job_satisfaction],
            "MaritalStatus": [marital_status], "MonthlyIncome": [monthly_income], "MonthlyRate": [monthly_rate],
            "NumCompaniesWorked": [num_companies_worked], "OverTime": [overtime], "PercentSalaryHike": [percent_salary_hike],
            "PerformanceRating": [performance_rating], "RelationshipSatisfaction": [relationship_satisfaction],
            "StockOptionLevel": [stock_option_level], "TotalWorkingYears": [total_working_years],
            "TrainingTimesLastYear": [training_times_last_year], "WorkLifeBalance": [work_life_balance],
            "YearsAtCompany": [years_at_company], "YearsInCurrentRole": [years_in_current_role],
            "YearsSinceLastPromotion": [years_since_last_promotion], "YearsWithCurrManager": [years_with_curr_manager]
        })

        try:
            probability = model.predict_proba(employee_data)[0][1]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # Dynamic Classification relative to slider threshold and high-risk line (0.70)
        if probability >= 0.70:
            risk_level, risk_class, risk_icon = "HIGH RISK", "risk-high", "⚠️"
            risk_message = "The employee has a high predicted probability of attrition."
            alert_class, alert_icon, alert_title = "high-alert", "⚠️", "High Risk"
            alert_message = "The predicted attrition probability is above 70%. This employee requires immediate attention or retention analysis."
        elif probability >= DECISION_THRESHOLD:
            risk_level, risk_class, risk_icon = "MEDIUM RISK", "risk-medium", "⚠️"
            risk_message = "The employee has crossed the active decision threshold."
            alert_class, alert_icon, alert_title = "medium-alert", "⚠️", "Medium Risk"
            alert_message = f"The probability exceeds your active threshold ({DECISION_THRESHOLD:.2f}). Consider reviewing engagement factors."
        else:
            risk_level, risk_class, risk_icon = "LOW RISK", "risk-low", "✅"
            risk_message = "The employee is below the decision threshold."
            alert_class, alert_icon, alert_title = "low-alert", "✅", "Low Risk"
            alert_message = f"The predicted attrition probability is below your active threshold ({DECISION_THRESHOLD:.2f}). Low risk indicated."

        st.markdown("---")
        st.html("""<div class="section-heading">🎯 Prediction Result</div>""")

        result_col1, result_col2 = st.columns(2, gap="large")

        with result_col1:
            st.html(f"""
            <div class="prediction-card">
                <div class="result-label">Employee Risk</div>
                <div class="{risk_class}">{risk_icon} {risk_level}</div>
                <div class="risk-message">{risk_message}</div>
                <div class="threshold-text">Active decision threshold: <strong>{DECISION_THRESHOLD:.2f}</strong></div>
            </div>
            """)

        with result_col2:
            st.html(f"""
            <div class="prediction-card">
                <div class="result-label">Attrition Probability</div>
                <div class="probability-value">{probability:.2%}</div>
                <div class="probability-description">Estimated probability that this employee may leave the organization.</div>
            </div>
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        st.html("""<div class="result-label">Risk Probability</div>""")
        st.progress(float(probability))

        st.html(f"""
        <div class="risk-alert {alert_class}">
            {alert_icon}<strong>{alert_title}:</strong> {alert_message}
        </div>
        """)

        st.markdown("---")
        st.html("""<div class="section-heading">🔍 Key Risk Drivers (Interpretability Insight)</div>""")
        st.info(
            "💡 **HR Actionable Insight:** Primary indicators like **OverTime status**, "
            "**Job Satisfaction**, and **Monthly Income** heavily sway this outcome. Focus retention efforts "
            "on addressing engagement metrics or compensation adjustments if flagged as high risk."
        )


# =========================================================
# TAB 2: BATCH CSV PREDICTION
# =========================================================

with tab2:
    st.html("""
    <div class="section-heading">
        📁 Batch Employee Attrition Analysis
    </div>
    <div class="section-description">
        Upload a CSV file containing multiple employee records matching the IBM HR dataset schema to run bulk predictions.
    </div>
    """)

    uploaded_file = st.file_uploader("Upload Employee CSV File", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded file with **{len(batch_df)}** records and **{len(batch_df.columns)}** features.")
            
            with st.expander("🔍 Preview Uploaded Dataset"):
                st.dataframe(batch_df.head())

            if st.button("🚀 Run Bulk Attrition Predictions", use_container_width=True, key="batch_btn"):
                with st.spinner("Processing batch predictions..."):
                    probabilities = model.predict_proba(batch_df)[:, 1]
                    
                    # Risk categories based on interactive threshold
                    risk_levels = []
                    for p in probabilities:
                        if p >= 0.70:
                            risk_levels.append("High Risk")
                        elif p >= DECISION_THRESHOLD:
                            risk_levels.append("Medium Risk")
                        else:
                            risk_levels.append("Low Risk")

                    results_df = batch_df.copy()
                    results_df["Attrition_Probability"] = probabilities
                    results_df["Attrition_Probability_Pct"] = [f"{p:.2%}" for p in probabilities]
                    results_df["Risk_Level"] = risk_levels

                st.markdown("---")
                st.html("""<div class="section-heading">📊 Batch Summary Metrics</div>""")
                
                m1, m2, m3 = st.columns(3)
                total_emp = len(results_df)
                high_risk_count = sum(results_df["Risk_Level"] == "High Risk")
                med_risk_count = sum(results_df["Risk_Level"] == "Medium Risk")
                low_risk_count = sum(results_df["Risk_Level"] == "Low Risk")

                m1.metric("Total Employees Analyzed", total_emp)
                m2.metric("High/Medium Risk", high_risk_count + med_risk_count, delta_color="inverse")
                m3.metric("Low Risk Employees", low_risk_count)

                st.markdown("<br>", unsafe_allow_html=True)
                st.html("""<div class="section-heading">📋 Prediction Results Table</div>""")
                st.dataframe(results_df[["Risk_Level", "Attrition_Probability_Pct"] + [col for col in batch_df.columns[:5]]])

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Results CSV",
                    data=csv_data,
                    file_name="employee_attrition_predictions_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_btn"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("📌 **Tip:** Make sure your CSV contains all 30 standard features from the IBM HR Analytics dataset.")


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    Employee Attrition Prediction
    &nbsp;•&nbsp;
    Python
    &nbsp;•&nbsp;
    Pandas
    &nbsp;•&nbsp;
    Scikit-learn
    &nbsp;•&nbsp;
    Logistic Regression
    &nbsp;•&nbsp;
    Streamlit
</div>
""")