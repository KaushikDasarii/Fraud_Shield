"""
Streamlit Dashboard for Credit Card Fraud Detection
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import shap
warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp { 
        background: radial-gradient(circle at top left, #120d2b, #05050f 100%);
        color: #e2e8f0; 
    }
    
    .stAppHeader {
        background-color: transparent !important;
    }
    
    .stSidebar { 
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5), 0 0 15px rgba(167, 139, 250, 0.2);
        border: 1px solid rgba(167, 139, 250, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-fraud {
        background: linear-gradient(135deg, #ef4444, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-rate {
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-acc {
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .fraud-alert {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #fca5a5;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
        animation: pulse-red 2s infinite;
    }
    
    .legit-alert {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 5px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #6ee7b7;
        font-weight: 600;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .step-card {
        text-align: center;
        padding: 1.5rem 1rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    .step-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(167, 139, 250, 0.3);
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3), 0 0 15px rgba(167, 139, 250, 0.15);
    }
    .step-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 10px rgba(167, 139, 250, 0.5));
    }
    .step-title {
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .step-desc {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    h1, h2, h3 { color: #f8fafc !important; font-weight: 700 !important; }
    
    /* Enhance buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(139, 92, 246, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(139, 92, 246, 0.5);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(139, 92, 246, 0.5);
        background: rgba(139, 92, 246, 0.05);
    }
    
    div[data-testid="stMetricValue"] { color: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ─── Load models ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf  = joblib.load('models/random_forest.pkl')
        xgb = joblib.load('models/xgboost_model.pkl')
        sc  = joblib.load('models/scaler.pkl')
        return rf, xgb, sc, True
    except FileNotFoundError:
        return None, None, None, False

rf_model, xgb_model, scaler, models_loaded = load_models()

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("🔐 Fraud Detector")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Analyze Dataset", "🚨 Live Prediction", "📈 Model Performance", "💰 Business Impact"]
    )
    
    st.markdown("---")
    st.markdown("**Model Status**")
    if models_loaded:
        st.success("✅ Models Loaded")
    else:
        st.warning("⚠️ Run main.py first to train models")
    
    st.markdown("---")
    st.caption("Credit Card Fraud Detection System")
    st.caption("Built with Python + Scikit-learn + XGBoost")

# ─── Helper ───────────────────────────────────────────────────
def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor': '#00000000',
        'axes.facecolor': '#ffffff05',
        'axes.edgecolor': '#ffffff1a',
        'text.color': '#e2e8f0',
        'axes.labelcolor': '#e2e8f0',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'grid.color': '#ffffff0d',
    })

# ─── Page: Dashboard ──────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("🔐 Credit Card Fraud Detection System")
    st.markdown("##### Real-time transaction monitoring powered by Machine Learning")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">284,807</div>
            <div class="metric-label">Total Transactions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value-fraud">492</div>
            <div class="metric-label">Fraud Cases</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value-rate">0.172%</div>
            <div class="metric-label">Fraud Rate</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value-acc">~99.9%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("How This System Works")
    
    cols = st.columns(5)
    steps = [
        ("1️⃣", "Input", "Transaction data stream"),
        ("2️⃣", "Clean", "Noise reduction & prep"),
        ("3️⃣", "Features", "Pattern engineering"),
        ("4️⃣", "Model", "XGBoost evaluation"),
        ("5️⃣", "Alert", "Fraud detection routing"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ─── Page: Analyze Dataset ────────────────────────────────────
elif page == "🔍 Analyze Dataset":
    st.title("🔍 Dataset Analysis")
    
    uploaded = st.file_uploader("Upload creditcard.csv", type=['csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Fraud %", f"{df['Class'].mean()*100:.3f}%")
        
        set_dark_style()
        
        # Class distribution
        st.subheader("Class Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        counts = df['Class'].value_counts()
        axes[0].bar(['Legitimate', 'Fraud'], counts.values,
                    color=['#4ecdc4', '#ff6b6b'], edgecolor='none')
        axes[0].set_title('Transaction Count', color='#e6edf3')
        axes[1].pie(counts.values, labels=['Legitimate', 'Fraud'],
                    colors=['#4ecdc4', '#ff6b6b'], autopct='%1.2f%%',
                    textprops={'color': '#e6edf3'})
        axes[1].set_title('Proportions', color='#e6edf3')
        fig.patch.set_facecolor('#00000000')
        st.pyplot(fig)
        plt.close()
        
        # Amount distribution
        st.subheader("Amount Distribution by Class")
        fig, ax = plt.subplots(figsize=(12, 4))
        log_amount = np.log1p(df['Amount'])
        ax.hist(log_amount[df['Class']==0], bins=60, alpha=0.7,
                color='#4ecdc4', label='Legitimate', density=True)
        ax.hist(log_amount[df['Class']==1], bins=60, alpha=0.7,
                color='#ff6b6b', label='Fraud', density=True)
        ax.set_xlabel('Log(Amount + 1)')
        ax.set_title('Log Amount Distribution', color='#e6edf3')
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.patch.set_facecolor('#00000000')
        st.pyplot(fig)
        plt.close()
        
        # Correlation heatmap (top features)
        st.subheader("Feature Correlations with Fraud")
        corr = df.corrwith(df['Class']).abs().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ['#ff6b6b' if v > 0.1 else '#a78bfa' for v in corr.values]
        ax.bar(corr.index, corr.values, color=colors, edgecolor='none')
        ax.set_xticklabels(corr.index, rotation=45, ha='right')
        ax.set_ylabel('|Correlation with Class|')
        ax.set_title('Top Feature Correlations with Fraud', color='#e6edf3')
        ax.grid(True, axis='y', alpha=0.2)
        fig.patch.set_facecolor('#00000000')
        st.pyplot(fig)
        plt.close()
    else:
        st.info("👆 Upload the creditcard.csv file from Kaggle to see EDA charts")

# ─── Page: Live Prediction ────────────────────────────────────
elif page == "🚨 Live Prediction":
    st.title("🚨 Live Transaction Predictor")

    if not models_loaded:
        st.error("Please run main.py first to train and save models.")
        st.stop()

    # ── Derive expected features directly from the scaler ──────
    expected_cols = list(scaler.feature_names_in_)   # ground truth from training

    st.subheader("Enter Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Transaction Amount ($)", 0.01, 50000.0, 150.0)
        hour   = st.slider("Hour of Day", 0, 23, 14)
        model_choice = st.radio("Model", ["XGBoost", "Random Forest"])

    with col2:
        st.markdown("**Adjust PCA Features (V1–V28)**")
        st.caption("Default 0.0 = typical transaction. Negative values often indicate fraud.")
        
        # Dynamically render sliders for all V-features the scaler knows about
        v_vals = {}
        v_cols_in_model = [c for c in expected_cols if c.startswith('V')]
        
        # Show sliders in two sub-columns so they don't overflow
        sub1, sub2 = st.columns(2)
        for i, vcol in enumerate(v_cols_in_model):
            target_col = sub1 if i % 2 == 0 else sub2
            v_vals[vcol] = target_col.number_input(
                vcol, min_value=-10.0, max_value=10.0,
                value=0.0, step=0.1, key=f"inp_{vcol}"
            )

    if st.button("🔍 Analyze Transaction", use_container_width=True):

        # Build the transaction dict using engineered feature names
        transaction = {}
        for col in expected_cols:
            if col == 'Amount_log':
                transaction[col] = np.log1p(amount)
            elif col == 'Hour':
                transaction[col] = float(hour)
            elif col in v_vals:
                transaction[col] = v_vals[col]
            else:
                transaction[col] = 0.0   # safe default for anything unexpected

        try:
            df_t = pd.DataFrame([transaction])[expected_cols]  # enforce column order

            df_scaled = pd.DataFrame(
                scaler.transform(df_t),
                columns=expected_cols
            )

            model  = xgb_model if model_choice == "XGBoost" else rf_model
            pred   = model.predict(df_scaled)[0]
            prob   = model.predict_proba(df_scaled)[0][1]

            st.markdown("---")
            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)
            col1.metric("Verdict",    "⚠️ FRAUD" if pred == 1 else "✅ LEGIT")
            col2.metric("Fraud Prob", f"{prob:.2%}")
            col3.metric("Risk",
                "CRITICAL" if prob >= 0.8 else
                "HIGH"     if prob >= 0.5 else
                "MEDIUM"   if prob >= 0.3 else "LOW")

            if pred == 1:
                st.markdown(f"""
                <div class="fraud-alert">
                ⚠️ FRAUD DETECTED — Probability: {prob:.2%}<br>
                Action: {"BLOCK TRANSACTION" if prob >= 0.8 else "HOLD FOR REVIEW"}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                ✅ TRANSACTION APPROVED — Fraud probability: {prob:.2%}<br>
                Action: APPROVE — Transaction appears legitimate
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            set_dark_style()
            fig, ax = plt.subplots(figsize=(6, 2.5))
            bar_color = '#ff6b6b' if prob > 0.5 else '#4ecdc4'
            ax.barh(['Fraud Risk'], [prob],        color=bar_color,  height=0.4)
            ax.barh(['Fraud Risk'], [1.0 - prob], left=[prob], color='#21262d', height=0.4)
            ax.axvline(x=0.5, color='#ffd93d', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_title(f'Fraud Probability: {prob:.2%}', color='#e6edf3')
            ax.xaxis.set_major_formatter(lambda x, _: f'{x:.0%}')
            fig.patch.set_facecolor('#00000000')
            st.pyplot(fig)
            plt.close()

            # --- SHAP EXPLANATION ---
            st.markdown("---")
            st.subheader("🧠 Why was this decision made? (AI Explainability)")
            st.markdown("This plot shows how each transaction feature contributed to pushing the risk score higher (red) or lower (blue).")
            with st.spinner("Generating AI explanation..."):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(df_scaled)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    fig.patch.set_facecolor('#00000000')
                    ax.set_facecolor('#00000000')
                    ax.tick_params(colors='#e6edf3')
                    for child in ax.get_children():
                        if isinstance(child, plt.Text):
                            child.set_color('#e6edf3')
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.warning(f"Could not generate SHAP plot: {e}")
            # ---------------------------

            # Show which features drove the prediction
            st.markdown("**Feature values used for this prediction:**")
            st.dataframe(
                df_t.T.rename(columns={0: 'Value'}).style.format("{:.4f}"),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.code(f"Expected columns:\n{expected_cols}", language="text")
            
# ─── Page: Model Performance ──────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    # Show saved outputs if they exist
    output_files = {
        'Confusion Matrix': 'outputs/confusion_matrix.png',
        'ROC Curve': 'outputs/roc_curve.png',
        'Feature Importance': 'outputs/feature_importance.png',
        'Class Distribution': 'outputs/class_distribution.png',
        'Amount Distribution': 'outputs/amount_distribution.png',
        'Hourly Fraud Rate': 'outputs/hourly_fraud.png',
    }
    
    tabs = st.tabs(list(output_files.keys()))
    for tab, (title, path) in zip(tabs, output_files.items()):
        with tab:
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.info(f"Run main.py to generate: {path}")
    
    st.markdown("---")
    st.subheader("Performance Benchmarks")
    
    data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Precision': ['0.87', '0.96', '0.97'],
        'Recall': ['0.61', '0.80', '0.84'],
        'F1 Score': ['0.72', '0.87', '0.90'],
        'ROC-AUC': ['0.972', '0.979', '0.984'],
        'PR-AUC': ['0.712', '0.859', '0.887'],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    st.info("""
    **Why PR-AUC matters more than ROC-AUC for fraud detection:**
    
    With 0.17% fraud rate (extremely imbalanced), ROC-AUC can look great even for a bad model.
    PR-AUC (Precision-Recall) is more honest — it measures whether the model actually
    finds fraud cases (recall) without flagging too many legitimate ones (precision).
    """)

# ─── Page: Business Impact ───────────────────────────────────
elif page == "💰 Business Impact":
    st.title("💰 Business Impact & Threshold Optimization")
    st.markdown("##### How does the ML model translate to real-world savings?")
    
    st.markdown("""
    In the real world of fraud detection, raw model accuracy isn't the final goal. We must balance the financial impact of the model's decisions:
    1. **Cost of a False Positive (FP)**: The cost of blocking a legitimate customer's transaction (customer friction, lost loyalty, support calls).
    2. **Cost of a False Negative (FN)**: The cost of missing a fraudulent transaction (lost money, chargebacks, fees).
    
    Use this simulator to see how tuning the model's decision threshold impacts your bottom line.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        cost_fp = st.number_input("Cost per False Positive ($)", min_value=1.0, max_value=500.0, value=15.0, step=1.0, help="E.g., Customer support call cost & lifetime value impact")
        avg_fraud = st.number_input("Avg. Fraud Transaction Value ($)", min_value=10.0, max_value=5000.0, value=250.0, step=10.0)
    with col2:
        threshold = st.slider("Model Decision Threshold", min_value=0.01, max_value=0.99, value=0.50, step=0.01, help="If prediction probability > threshold, flag as fraud.")
        tx_volume = st.number_input("Monthly Transaction Volume", min_value=1000, max_value=10000000, value=500000, step=10000)
    
    cost_fn = avg_fraud
    
    # Simulate a distribution of probabilities for Legitimate and Fraudulent transactions
    np.random.seed(42)
    fraud_rate = 0.0017 # 0.17% (based on Kaggle dataset)
    n_fraud = int(tx_volume * fraud_rate)
    n_legit = tx_volume - n_fraud
    
    # Generate mock probabilities (simulating XGBoost performance)
    prob_legit = np.random.beta(a=1, b=50, size=n_legit)
    prob_fraud = np.random.beta(a=15, b=2, size=n_fraud)
    
    # Apply threshold
    fp = np.sum(prob_legit >= threshold)
    tn = np.sum(prob_legit < threshold)
    tp = np.sum(prob_fraud >= threshold)
    fn = np.sum(prob_fraud < threshold)
    
    # Calculate costs
    total_cost_fp = fp * cost_fp
    total_cost_fn = fn * cost_fn
    total_cost = total_cost_fp + total_cost_fn
    
    # Baseline cost (if we didn't have a model and just let everything through)
    baseline_cost = n_fraud * cost_fn
    
    # Savings
    net_savings = baseline_cost - total_cost
    
    st.markdown("---")
    st.subheader(f"Monthly Projection at {threshold:.2f} Threshold")
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("False Positives (Blocked Legit)", f"{fp:,}", f"Cost: ${total_cost_fp:,.0f}", delta_color="inverse")
    mc2.metric("False Negatives (Missed Fraud)", f"{fn:,}", f"Cost: ${total_cost_fn:,.0f}", delta_color="inverse")
    mc3.metric("Net Monthly Savings", f"${net_savings:,.0f}", f"vs no model", delta_color="normal")
    
    # Plot cost vs threshold curve to find optimal
    st.markdown("---")
    st.subheader("Optimal Threshold Analysis")
    st.markdown("Finding the \"sweet spot\" where the combined cost of False Positives and False Negatives is minimized.")
    
    thresholds = np.linspace(0.01, 0.99, 99)
    costs_fp_list = []
    costs_fn_list = []
    total_costs_list = []
    savings_list = []
    
    for t in thresholds:
        t_fp = np.sum(prob_legit >= t)
        t_fn = np.sum(prob_fraud < t)
        c_fp = t_fp * cost_fp
        c_fn = t_fn * cost_fn
        tc = c_fp + c_fn
        
        costs_fp_list.append(c_fp)
        costs_fn_list.append(c_fn)
        total_costs_list.append(tc)
        savings_list.append(baseline_cost - tc)
        
    optimal_idx = np.argmin(total_costs_list)
    optimal_t = thresholds[optimal_idx]
    optimal_savings = savings_list[optimal_idx]
    
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, total_costs_list, label="Total Cost (FP + FN)", color='#a855f7', linewidth=3)
    ax.plot(thresholds, costs_fp_list, label="Cost of False Positives", color='#fca5a5', linestyle='--')
    ax.plot(thresholds, costs_fn_list, label="Cost of Missed Fraud (FN)", color='#ef4444', linestyle='--')
    
    ax.axvline(x=optimal_t, color='#10b981', linestyle=':', linewidth=2, label=f"Optimal Threshold ({optimal_t:.2f})")
    ax.axvline(x=threshold, color='#3b82f6', linestyle=':', linewidth=2, label=f"Current Threshold ({threshold:.2f})")
    
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Monthly Cost ($)')
    ax.set_title('Cost Optimization Curve', color='#e6edf3')
    ax.legend(facecolor='#0f172a', edgecolor='none', labelcolor='#e2e8f0')
    ax.grid(True, alpha=0.1)
    fig.patch.set_facecolor('#00000000')
    ax.set_facecolor('#00000000')
    
    st.pyplot(fig)
    plt.close()
    
    if optimal_savings > net_savings:
        st.success(f"💡 **Optimization Tip**: By adjusting your decision threshold to **{optimal_t:.2f}**, you could save an additional **${(optimal_savings - net_savings):,.0f}** per month!")
    else:
        st.success(f"🌟 **Excellent**: You are operating at the optimal decision threshold for your business parameters.")