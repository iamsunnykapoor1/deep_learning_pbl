"""
================================================================================
Deep Learning-Based Fertilizer Recommendation for Sustainable Crop Production
Script: app.py
Description: Interactive Streamlit Web Application for Deep Learning Fertilizer
             Recommendation with dedicated Submit / Predict buttons, presets,
             detailed agronomic explanations, and evaluation metrics.
================================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import inference functions
from predict import predict_fertilizer, get_supported_options, FERTILIZER_INFO, MODELS_DIR

# Set Streamlit page configuration
st.set_page_config(
    page_title="Deep Learning Fertilizer Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for agricultural UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1e4620;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #437057;
        text-align: center;
        margin-bottom: 20px;
    }
    .rec-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #2e7d32;
        padding: 22px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .rec-title {
        color: #1b5e20;
        font-size: 1.9rem;
        font-weight: bold;
        margin-bottom: 2px;
    }
    .rec-grade {
        color: #2e7d32;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .confidence-badge {
        background-color: #2e7d32;
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    div.stButton > button:first-child {
        background-color: #2e7d32;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child:hover {
        background-color: #1b5e20;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Preset sample test cases for rapid testing
PRESETS = {
    "🌱 Urea (High Nitrogen)": {
        "soil": "Sandy", "crop": "Maize", "temp": 26.0, "hum": 52.0, "moist": 38.0, "n": 37.0, "k": 0.0, "p": 0.0
    },
    "🌾 DAP (High Phosphorus)": {
        "soil": "Clayey", "crop": "Pulses", "temp": 27.0, "hum": 54.0, "moist": 28.0, "n": 13.0, "k": 0.0, "p": 40.0
    },
    "🌿 14-35-14 (High P Complex)": {
        "soil": "Black", "crop": "Cotton", "temp": 34.0, "hum": 65.0, "moist": 62.0, "n": 7.0, "k": 9.0, "p": 30.0
    },
    "🍀 28-28 (Balanced N & P)": {
        "soil": "Red", "crop": "Tobacco", "temp": 32.0, "hum": 62.0, "moist": 34.0, "n": 22.0, "k": 0.0, "p": 20.0
    },
    "🍂 10-26-26 (High P & K)": {
        "soil": "Loamy", "crop": "Wheat", "temp": 28.0, "hum": 54.0, "moist": 35.0, "n": 10.0, "k": 19.0, "p": 20.0
    },
    "🍁 17-17-17 (Equal N-P-K)": {
        "soil": "Sandy", "crop": "Barley", "temp": 26.0, "hum": 52.0, "moist": 35.0, "n": 12.0, "k": 10.0, "p": 13.0
    },
    "🍃 20-20 (Moderate N & P)": {
        "soil": "Red", "crop": "Cotton", "temp": 25.0, "hum": 50.0, "moist": 64.0, "n": 9.0, "k": 0.0, "p": 10.0
    }
}


def load_preset_callback(preset_key):
    preset = PRESETS[preset_key]
    st.session_state["w_soil_type"] = preset["soil"]
    st.session_state["w_crop_type"] = preset["crop"]
    st.session_state["w_temperature"] = preset["temp"]
    st.session_state["w_humidity"] = preset["hum"]
    st.session_state["w_moisture"] = preset["moist"]
    st.session_state["w_nitrogen"] = preset["n"]
    st.session_state["w_potassium"] = preset["k"]
    st.session_state["w_phosphorous"] = preset["p"]
    st.session_state["has_predicted"] = True


def main():
    st.markdown('<div class="main-header">🌱 Deep Learning Fertilizer Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Precision Agriculture Decision Support using Artificial Neural Networks (ANN)</div>', unsafe_allow_html=True)

    soil_options, crop_options = get_supported_options()

    # Initialize session state with default values if not present
    if "w_soil_type" not in st.session_state:
        st.session_state["w_soil_type"] = "Sandy"
        st.session_state["w_crop_type"] = "Maize"
        st.session_state["w_temperature"] = 26.0
        st.session_state["w_humidity"] = 52.0
        st.session_state["w_moisture"] = 38.0
        st.session_state["w_nitrogen"] = 37.0
        st.session_state["w_potassium"] = 0.0
        st.session_state["w_phosphorous"] = 0.0
        st.session_state["has_predicted"] = True

    # SIDEBAR: Quick Presets
    st.sidebar.header("⚡ 1-Click Scenario Presets")
    st.sidebar.markdown("Click any preset to load conditions:")
    
    preset_keys = list(PRESETS.keys())
    col_p1, col_p2 = st.sidebar.columns(2)
    for i, p_key in enumerate(preset_keys):
        col = col_p1 if i % 2 == 0 else col_p2
        col.button(p_key, key=f"btn_p_{i}", on_click=load_preset_callback, args=(p_key,), use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.header("📋 Input Parameters")

    # 1. Soil & Crop Dropdowns
    soil_type = st.sidebar.selectbox(
        "Soil Type",
        options=soil_options,
        index=soil_options.index(st.session_state["w_soil_type"]) if st.session_state["w_soil_type"] in soil_options else 0,
        key="w_soil_type"
    )
    crop_type = st.sidebar.selectbox(
        "Crop Type",
        options=crop_options,
        index=crop_options.index(st.session_state["w_crop_type"]) if st.session_state["w_crop_type"] in crop_options else 0,
        key="w_crop_type"
    )

    # 2. Climate & Soil Moisture Sliders
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        temperature = st.sidebar.slider("Temperature (°C)", min_value=15.0, max_value=45.0, step=0.5, key="w_temperature")
    with col_s2:
        humidity = st.sidebar.slider("Humidity (%)", min_value=30.0, max_value=90.0, step=1.0, key="w_humidity")

    moisture = st.sidebar.slider("Soil Moisture (%)", min_value=10.0, max_value=85.0, value=float(st.session_state["w_moisture"]), step=1.0, key="w_moisture")

    # 3. Soil Nutrients (N-P-K)
    st.sidebar.markdown("### 🧪 Soil Nutrients (N - K - P)")
    col_n1, col_n2, col_n3 = st.sidebar.columns(3)
    with col_n1:
        nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=1.0, key="w_nitrogen")
    with col_n2:
        potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=1.0, key="w_potassium")
    with col_n3:
        phosphorous = st.sidebar.number_input("Phosphorous (P)", min_value=0.0, max_value=100.0, step=1.0, key="w_phosphorous")

    # SUBMIT / PREDICT BUTTON IN SIDEBAR
    st.sidebar.markdown("---")
    submit_sidebar = st.sidebar.button("🌿 Recommend Fertilizer (Submit)", type="primary", use_container_width=True)

    if submit_sidebar:
        st.session_state["has_predicted"] = True

    # MAIN TABS
    tab_rec, tab_guide, tab_model, tab_data = st.tabs([
        "🌾 Fertilizer Recommendation", 
        "📖 Fertilizer Action Guide",
        "📊 Neural Network & Metrics", 
        "📁 Dataset Overview"
    ])

    # TAB 1: RECOMMENDATION
    with tab_rec:
        st.markdown("### 🌾 Recommendation Result")

        # Perform Deep Learning Inference
        try:
            result = predict_fertilizer(
                temperature=temperature,
                humidity=humidity,
                moisture=moisture,
                soil_type=soil_type,
                crop_type=crop_type,
                nitrogen=nitrogen,
                potassium=potassium,
                phosphorous=phosphorous
            )

            rec_name = result["recommended_fertilizer"]
            confidence = result["confidence"]
            details = result["details"]
            probs = result["probabilities"]
            dosage = result["dosage"]

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Tip: Run 'python train_model.py' to generate model artifacts.")
            return

        # Result Header Box
        st.markdown(f"""
        <div class="rec-box">
            <div class="rec-title">Recommended Fertilizer: {rec_name}</div>
            <div class="rec-grade">Grade / Formulation: {details.get('grade', 'Standard')}</div>
            <div class="confidence-badge">Recommendation Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🎯 Recommended Rate for Resource Efficiency")
        dosage_col1, dosage_col2, dosage_col3 = st.columns(3)
        dosage_col1.metric("Rate per acre", f"{dosage['dosage_kg_acre']:.1f} kg")
        dosage_col2.metric("Rate per hectare", f"{dosage['dosage_kg_ha']:.1f} kg")
        dosage_col3.metric("50 kg bags per acre", f"{dosage['bags_50kg_per_acre']:.2f}")
        st.caption(
            f"Estimated soil nutrient sufficiency: {dosage['nutrient_sufficiency_percent']:.1f}%. "
            "The rate is adjusted for crop demand, soil retention, and measured N-P-K levels."
        )
        st.markdown(f"**Application method:** {dosage['application_method']}  ")
        st.markdown(f"**Split schedule:** {dosage['split_schedule']}")

        # What it will do details
        st.subheader(f"💡 What {rec_name} Will Do for Your Crop:")
        
        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.markdown(f"""
            **1. Primary Function in Crop:**
            > {details.get('what_it_does', details.get('description'))}
            
            **2. Soil & Plant Absorption Mechanism:**
            > {details.get('soil_action', 'Provides direct nutrient availability for root uptake.')}
            """)
        with c_a2:
            st.markdown(f"""
            **3. Optimal Application Timing:**
            > {details.get('best_timing', 'Apply during active growth phase.')}
            
            **4. Recommended Target Crops:**
            > {details.get('suitable_crops', 'Commercial and cereal crops.')}
            """)

        st.markdown("---")

        # Details Grid
        c1, c2 = st.columns([1.1, 0.9])

        with c1:
            st.subheader("📊 Combined Recommendation Scores")
            prob_df = pd.DataFrame({
                "Fertilizer": list(probs.keys()),
                "Probability (%)": list(probs.values())
            })
            
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            colors = ['#2e7d32' if f == rec_name else '#a5d6a7' for f in prob_df['Fertilizer']]
            bars = ax.barh(prob_df['Fertilizer'][::-1], prob_df['Probability (%)'][::-1], color=colors[::-1])
            ax.set_xlabel("Predicted Probability (%)", fontsize=10, fontweight='bold')
            ax.set_xlim(0, 105)
            for bar in bars:
                w = bar.get_width()
                if w >= 1:
                    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold', color='#1b5e20')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            st.subheader("📝 Active Field & Soil Conditions")
            summary_df = pd.DataFrame({
                "Parameter": ["Soil Type", "Crop Type", "Temperature", "Humidity", "Soil Moisture", "Nitrogen (N)", "Potassium (K)", "Phosphorous (P)"],
                "Value": [soil_type, crop_type, f"{temperature} °C", f"{humidity} %", f"{moisture} %", f"{nitrogen} mg/kg", f"{potassium} mg/kg", f"{phosphorous} mg/kg"]
            })
            st.table(summary_df)

            st.markdown("### 🧪 Current Nutrient Balance")
            total_npk = max(1.0, nitrogen + potassium + phosphorous)
            n_pct = (nitrogen / total_npk) * 100
            k_pct = (potassium / total_npk) * 100
            p_pct = (phosphorous / total_npk) * 100
            
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("N Ratio", f"{n_pct:.1f}%")
            col_b2.metric("K Ratio", f"{k_pct:.1f}%")
            col_b3.metric("P Ratio", f"{p_pct:.1f}%")

    # TAB 2: FERTILIZER ACTION GUIDE
    with tab_guide:
        st.subheader("📖 Complete Fertilizer Action & Function Guide")
        st.markdown("Detailed breakdown of what each recommended fertilizer does, its chemical composition, and when to use it:")

        for f_name, f_info in FERTILIZER_INFO.items():
            with st.expander(f"📌 {f_name} — {f_info['grade']}", expanded=(f_name == rec_name)):
                st.markdown(f"""
                * **What it will do:** {f_info['what_it_does']}
                * **Soil & Plant Action:** {f_info['soil_action']}
                * **Application Timing:** {f_info['best_timing']}
                * **Best Suited Crops:** {f_info['suitable_crops']}
                """)

    # TAB 3: NEURAL NETWORK & METRICS
    with tab_model:
        st.subheader("🧠 Deep Learning Architecture & Training Performance")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model Architecture", "Feed-Forward ANN (MLP)")
        m2.metric("Optimizer", "Adam (lr=0.01)")
        m3.metric("Loss Function", "Sparse Categorical Cross-Entropy")
        m4.metric("Input Dimensions", "22 Transformed Features")

        eval_plot_path = os.path.join(MODELS_DIR, "training_evaluation.png")
        if os.path.exists(eval_plot_path):
            st.image(eval_plot_path, caption="Training History (Accuracy & Loss Curves) and Test Set Confusion Matrix", use_container_width=True)
        else:
            st.warning("Training plots not found. Run 'python train_model.py' to generate.")

        st.markdown("""
        ### Why an Artificial Neural Network (ANN)?
        1. **Non-Linear Decision Boundaries**: Soil nutrient interactions and crop responses are inherently non-linear. The ReLU activations allow the network to learn intricate multi-variable relationships.
        2. **Dropout Regularization**: A 20% Dropout rate prevents the network from memorizing training samples and enhances generalizability.
        3. **Softmax Probabilistic Outputs**: Softmax normalization yields transparent confidence scores across all candidate fertilizer types.
        """)

    # TAB 4: DATASET OVERVIEW
    with tab_data:
        st.subheader("📁 Fertilizer Prediction Benchmark Dataset")
        dataset_file = os.path.join(os.path.dirname(__file__), "dataset", "Fertilizer_Prediction.csv")
        
        if os.path.exists(dataset_file):
            df_preview = pd.read_csv(dataset_file)
            st.write(f"Total Records: **{df_preview.shape[0]}**, Features: **{df_preview.shape[1]-1}**, Target: **{df_preview.columns[-1]}**")
            st.dataframe(df_preview.head(15), use_container_width=True)
            
            st.subheader("🎯 Fertilizer Distribution in Dataset")
            fig_dist, ax_dist = plt.subplots(figsize=(8, 3.5))
            sns.countplot(data=df_preview, x=df_preview.columns[-1], palette='crest', ax=ax_dist)
            ax_dist.set_title("Distribution of Fertilizer Classes", fontsize=11, fontweight='bold')
            ax_dist.set_xlabel("Fertilizer Name")
            ax_dist.set_ylabel("Count")
            plt.xticks(rotation=30)
            st.pyplot(fig_dist)
        else:
            st.warning("Dataset file not found at dataset/Fertilizer_Prediction.csv")


if __name__ == "__main__":
    main()
