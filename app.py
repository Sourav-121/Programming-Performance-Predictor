import streamlit as st
import joblib
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="EDM Decision Support System", layout="wide")

# --- Helper Function for SHAP ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height if height else 400, scrolling=True)

# --- Title & Styling ---
st.title("🎓 Intelligent Student Performance Predictor")
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    .prediction-card {
        padding: 20px; 
        border-radius: 12px; 
        text-align: center; 
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h5 {
        padding-top: 15px;
        padding-bottom: 5px;
        color: #333;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("⚙️ Configuration")
    gender = st.radio("Select Student Gender", ["Male", "Female"])
    st.info(f"Analysis Mode: **{gender} Students**")
    if gender == "Female":
        st.caption("Includes: Weekly Study Time & Library Time")
    
    st.divider()
    st.write("Developed for EDM Research.")

# --- Load Models & Data ---
try:
    if gender == "Male":
        model = joblib.load("male_shap_model.pkl")
        features = joblib.load("male_shap_features.pkl")
    else:
        model = joblib.load("female_shap_model.pkl")
        features = joblib.load("female_shap_features.pkl")
except FileNotFoundError:
    st.error("⚠️ Model files missing! Please run 'train_model_shap.py' first.")
    st.stop()

# --- Conversion Functions ---
def map_online_problem_solved(value):
    if 0 <= value <= 299:
        return 1
    elif 300 <= value <= 799:
        return 2
    elif 800 <= value <= 1199:
        return 3
    elif 1200 <= value <= 1999:
        return 4
    elif value >= 2000:
        return 5

def map_stl_skills(value):
    mapping = {"Very Poor": 1, "Poor": 2, "Average": 3, "Good": 4, "Excellent": 5}
    return mapping.get(value, 3)  # Default to Average if not found

def map_problem_solving(value):
    mapping = {"Poor": 1, "Average": 2, "Good": 3, "Very Good": 4, "Excellent": 5}
    return mapping.get(value, 3)  # Default to Good if not found

def map_solved_per_onsite(value):
    if 0 <= value <= 1:
        return 1
    elif 2 <= value <= 3:
        return 3
    elif 4 <= value <= 5:
        return 3
    elif 6 <= value <= 8:
        return 4
    elif value >= 9:
        return 5

def map_onsite_participation(value):
    if 0 <= value <= 5:
        return 1
    elif 6 <= value <= 15:
        return 2
    elif 16 <= value <= 30:
        return 3
    elif 31 <= value <= 45:
        return 4
    elif value >= 46:
        return 5

def map_weekly_study_hours(value):
    if 0 <= value <= 4:
        return 1
    elif 5 <= value <= 9:
        return 2
    elif 10 <= value <= 14:
        return 3
    elif 15 <= value <= 20:
        return 4
    elif value >= 21:
        return 5

def map_weekly_library_visit(value):
    if 0 <= value <= 1:
        return 1
    elif 2 <= value <= 3:
        return 2
    elif 4 <= value <= 6:
        return 3
    elif 7 <= value <= 10:
        return 4
    elif value >= 11:
        return 5

def map_weekly_practice(value):
    mapping = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
    return mapping.get(value, 3)  # Default to Sometimes if not found

# --- Female Conversion Function for 'Learning From mistakes' ---
def map_learning_from_mistakes(value):
    mapping = {
        "Never": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Often": 4,
        "Always": 5
    }
    return mapping.get(value, 3)  # Default to 'Sometimes' if not found

# --- Main Layout ---
col1, col2 = st.columns([1, 2], gap="large")

# --- Column 1: Input Form ---
with col1:
    st.subheader("📝 Student Data Input")
    with st.form("input_form"):
        user_input = {}

        # **Male Input**: Does NOT include weekly study time and library time.
        if gender == "Male":
            # Online Problems Solved (Text Field)
            user_input['Online Problems Solved'] = map_online_problem_solved(int(st.text_input("Online Problems Solved", "0")))
            
            # STL Skill, Data Structure Knowledge, Algorithm Knowledge, etc.
            user_input['STL Skill'] = map_stl_skills(st.selectbox("STL Skill", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['Data Structure Knowledge'] = map_stl_skills(st.selectbox("Data Structure Knowledge", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['Algorithm Knowledge'] = map_stl_skills(st.selectbox("Algorithm Knowledge", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['Weekly Programming Practice'] = map_stl_skills(st.selectbox("Weekly Programming Practice", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['C Programming Skill'] = map_stl_skills(st.selectbox("C Programming Skill", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['OOP Skill'] = map_stl_skills(st.selectbox("OOP Skill", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            
            # Problem Solving Capability, Debugging Skills, etc.
            user_input['Problem Solving Capability'] = map_problem_solving(st.selectbox("Problem Solving Capability", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Debugging Skills'] = map_problem_solving(st.selectbox("Debugging Skills", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Problem Understanding Capability'] = map_problem_solving(st.selectbox("Problem Understanding Capability", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Complexity Analysis Capability'] = map_problem_solving(st.selectbox("Complexity Analysis Capability", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Critical Thinking Level'] = map_problem_solving(st.selectbox("Critical Thinking Level", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Creative Thinking Level'] = map_problem_solving(st.selectbox("Creative Thinking Level", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            
            # Solved per Onsite, Onsite Contest Participation (Text Fields)
            user_input['Solved per Onsite'] = map_solved_per_onsite(int(st.text_input("Solved per Onsite", "0")))
            user_input['Onsite Contest Participation'] = map_onsite_participation(int(st.text_input("Onsite Contest Participation", "0")))

        # **Female Input**: Includes weekly study time and library time.
        if gender == "Female":
            # Online Problems Solved (Text Field)
            user_input['Online Problems Solved'] = map_online_problem_solved(int(st.text_input("Online Problems Solved", "0")))
            
            # STL Skill, Data Structure Knowledge, Algorithm Knowledge, etc.
            user_input['STL Skill'] = map_stl_skills(st.selectbox("STL Skill", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['Data Structure Knowledge'] = map_stl_skills(st.selectbox("Data Structure Knowledge", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['Algorithm Knowledge'] = map_stl_skills(st.selectbox("Algorithm Knowledge", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['Weekly Programming Practice'] = map_stl_skills(st.selectbox("Weekly Programming Practice", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['C Programming Skill'] = map_stl_skills(st.selectbox("C Programming Skill", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            user_input['OOP Skill'] = map_stl_skills(st.selectbox("OOP Skill", ["Very Poor", "Poor", "Average", "Good", "Excellent"]))
            
            # Problem Solving Capability, Debugging Skills, etc.
            user_input['Problem Solving Capability'] = map_problem_solving(st.selectbox("Problem Solving Capability", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Debugging Skills'] = map_problem_solving(st.selectbox("Debugging Skills", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Problem Understanding Capability'] = map_problem_solving(st.selectbox("Problem Understanding Capability", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Complexity Analysis Capability'] = map_problem_solving(st.selectbox("Complexity Analysis Capability", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Critical Thinking Level'] = map_problem_solving(st.selectbox("Critical Thinking Level", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            user_input['Creative Thinking Level'] = map_problem_solving(st.selectbox("Creative Thinking Level", ["Poor", "Average", "Good", "Very Good", "Excellent"]))
            
            # Solved per Onsite, Onsite Contest Participation (Text Fields)
            user_input['Solved per Onsite'] = map_solved_per_onsite(int(st.text_input("Solved per Onsite", "0")))
            user_input['Onsite Contest Participation'] = map_onsite_participation(int(st.text_input("Onsite Contest Participation", "0")))
            
            # Weekly Study Hours, Weekly Library Visit Time (Text Fields)
            user_input['Weekly Study Time'] = map_weekly_study_hours(int(st.text_input("Weekly Study Time (Hours)", "0")))
            user_input['Weekly Library Time'] = map_weekly_library_visit(int(st.text_input("Weekly Library Time (Hours)", "0")))

            # Learning From Mistakes (New Mapping)
            user_input['Learning From mistakes'] = map_learning_from_mistakes(st.selectbox("Learning From Mistakes", ["Never", "Rarely", "Sometimes", "Often", "Always"]))

        submit = st.form_submit_button("Analyze Performance 🚀", type="primary")

# --- Column 2: Dashboard (Vertical Layout) ---
if submit:
    # Prepare Data
    input_df = pd.DataFrame([user_input])

    # --- Trim Column Names: Remove Leading/Trailing Spaces ---
    input_df.columns = input_df.columns.str.strip()  # Trim spaces from column names

    # Ensure columns match the trained model features (add space if needed to match exactly)
    try:
        # Adding the space to match exactly with the model's features
        input_df.columns = [col + " " if col == "Solved per Onsite" else col for col in input_df.columns]

        # Reorder the input_df to match the model features
        input_df = input_df[features]  # Reorder the input_df to match feature order
    except KeyError as e:
        st.error(f"KeyError: {e}. Ensure all features are correctly named and present.")
        st.stop()

    # Prediction Logic
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    max_prob = max(probs)
    
    # Color Map
    color_map = {"Excellent": "#28a745", "Good": "#17a2b8", "Average": "#ffc107", "Weak": "#dc3545"}
    pred_color = color_map.get(prediction, "#6c757d")

    with col2:
        # --- 1. Prediction Card (Top) ---
        st.subheader("📊 Analysis Report")
        st.markdown(f"""
        <div class="prediction-card" style="background-color: {pred_color}25; border: 2px solid {pred_color};">
            <h2 style="color: {pred_color}; margin:0;">{prediction}</h2>
            <p style="font-size: 18px; margin:0;">Confidence Level: <b>{max_prob*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- 2. Probability Bar Chart ---
        st.markdown("##### 1️⃣ Model Confidence Distribution")
        prob_df = pd.DataFrame({'Category': classes, 'Probability': probs})
        fig_bar = px.bar(prob_df, x='Category', y='Probability', 
                         color='Category', text_auto='.1%',
                         color_discrete_map=color_map)
        fig_bar.update_layout(showlegend=False, height=250, margin=dict(t=10, b=10, l=0, r=0))
        st.plotly_chart(fig_bar, use_container_width=True)
            
        # --- 3. Skill Radar Chart ---
        st.markdown("##### 2️⃣ Skill & Feature Profile (Radar View)")
        radar_vals = [min(v, 5) for k, v in user_input.items()]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals,
            theta=list(user_input.keys()),
            fill='toself',
            name='Student Profile',
            line_color=pred_color
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            height=400,
            margin=dict(t=30, b=30, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # --- 4. SHAP Force Plot ---
        st.markdown("##### 3️⃣ Feature Impact Analysis (SHAP)")
        st.caption("Red bars push the prediction towards the result, Blue bars push away.")
        
        # SHAP Calculation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        class_idx = list(classes).index(prediction)
        
        # Fix for Dimension Handling
        if isinstance(shap_values, list):
            shap_val_target = shap_values[class_idx][0]
        else:
            if len(shap_values.shape) == 3:
                shap_val_target = shap_values[0, :, class_idx]
            else:
                shap_val_target = shap_values[0]

        # Base Value Extraction
        if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
            base_val = explainer.expected_value[class_idx]
        else:
            base_val = explainer.expected_value

        # Generate Plot
        force_plot = shap.force_plot(
            base_val,
            shap_val_target,
            input_df.iloc[0].values,
            feature_names=input_df.columns.tolist(),
            matplotlib=False
        )
        st_shap(force_plot)

else:
    # Initial State
    with col2:
        st.info("👈 Please enter student data on the left to generate the analysis.")
        st.image("https://cdn-icons-png.flaticon.com/512/3079/3079165.png", width=150)
