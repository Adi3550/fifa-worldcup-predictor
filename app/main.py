import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

st.set_page_config(page_title="FIFA World Cup 2026 Predictor", layout="wide", page_icon="🏆")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .stMetric * {color: #000000 !important;}
    h1 {color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #2563EB;}
    </style>
""", unsafe_allow_html=True)

st.title("🏆 FIFA World Cup 2026 Predictor Pro")

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORICAL_PATH = os.path.join(BASE_DIR, "data", "raw", "historical_matches.csv")
STATS_PATH = os.path.join(BASE_DIR, "data", "raw", "supplementary_stats.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "final_features.csv")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "processed", "top_predictions.csv")

MODEL_DIR = os.path.join(BASE_DIR, "notebooks")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_models():
    if os.path.exists(RF_MODEL_PATH) and os.path.exists(SCALER_PATH):
        rf = joblib.load(RF_MODEL_PATH)
        lr = joblib.load(LR_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return rf, lr, scaler
    return None, None, None

# Load Datasets
df_hist = load_data(HISTORICAL_PATH)
df_stats = load_data(STATS_PATH)
df_proc = load_data(PROCESSED_PATH)
df_preds = load_data(PREDICTIONS_PATH)

rf_model, lr_model, scaler = load_models()

# Create Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Project Overview", 
    "📊 Data Explorer", 
    "📈 Model Performance Metrics", 
    "🧠 Feature Importance",
    "🔮 Final Predictions"
])

with tab1:
    st.header("Project Overview")
    st.write("Welcome to the FIFA World Cup 2026 Predictor. This project predicts tournament success based on historical data and engineered squad metrics.")
    st.write("- **Historical Matches:** Analyzes past international matches.")
    st.write("- **Squad Demographics:** Factors in average age, caps, and FIFA rankings.")

with tab2:
    st.header("Dataset Overview & Comparison")
    st.write("Compare the raw data sources against the final engineered feature set.")
    
    data_sel = st.radio("Select Dataset to View:", ["Historical Matches (Raw)", "Team Statistics (Raw)", "Engineered Features (Processed)"], horizontal=True)
    
    if data_sel == "Historical Matches (Raw)":
        if df_hist is not None:
            st.dataframe(df_hist, use_container_width=True)
            st.caption(f"Total historical matches: {len(df_hist)}")
        else:
            st.error("Historical data not found.")
            
    elif data_sel == "Team Statistics (Raw)":
        if df_stats is not None:
            st.dataframe(df_stats, use_container_width=True)
            st.caption("2026 Qualified Teams Supplementary Data")
        else:
            st.error("Stats data not found.")
            
    elif data_sel == "Engineered Features (Processed)":
        if df_proc is not None:
            st.dataframe(df_proc, use_container_width=True)
            st.caption("Final merged dataset with target variables ready for ML training.")

with tab3:
    st.header("📈 Model Performance Metrics")
    st.write("Full transparency into the machine learning pipeline's decision boundaries and performance metrics.")
    
    if df_proc is not None and rf_model is not None:
        X_test_path = os.path.join(MODEL_DIR, "X_test.csv")
        y_test_path = os.path.join(MODEL_DIR, "y_test.csv")
        
        if os.path.exists(X_test_path) and os.path.exists(y_test_path):
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).squeeze()
            X_test_scaled = scaler.transform(X_test)
        else:
            st.error("Test sets not found. Please run the training notebooks.")
            st.stop()
        
        # Get predictions
        rf_preds = rf_model.predict(X_test_scaled)
        rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
        lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Display Metrics
        st.subheader("Performance Metrics (Test Set)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest Accuracy", f"{accuracy_score(y_test, rf_preds):.1%}")
        m2.metric("Random Forest ROC-AUC", f"{roc_auc_score(y_test, rf_probs):.3f}")
        m3.metric("Logistic Regression ROC-AUC", f"{roc_auc_score(y_test, lr_probs):.3f}")
        
        st.divider()
        col_charts1, col_charts2 = st.columns(2)
        
        with col_charts1:
            st.subheader("ROC Curve Comparison")
            st.markdown("*The **ROC Curve** shows how well the models distinguish between winners and losers. A curve closer to the top-left corner (and a higher AUC score) means the model is highly accurate.*")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
            ax_roc.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.2f})", color='green')
            ax_roc.plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC = {roc_auc_score(y_test, lr_probs):.2f})", color='blue')
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend()
            st.pyplot(fig_roc)
            
        with col_charts2:
            st.subheader("Random Forest Confusion Matrix")
            st.markdown("*The **Confusion Matrix** shows prediction accuracy. The top-left and bottom-right squares show correct predictions. The other squares represent errors (False Positives/Negatives).*")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            cm = confusion_matrix(y_test, rf_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                        xticklabels=['No Win', 'Home Win'], yticklabels=['No Win', 'Home Win'])
            ax_cm.set_ylabel('Actual Outcome')
            ax_cm.set_xlabel('Predicted Outcome')
            st.pyplot(fig_cm)
            
        st.divider()
        st.info("💡 **Interpretation:** In sports forecasting, a **False Positive** (predicting a team will win when they don't) damages the model's credibility. A **False Negative** (predicting a team will lose but they pull off an underdog victory) is statistically expected because of the unpredictable nature of sports. Therefore, models with high Precision (fewer False Positives) are generally preferred.")
            
    else:
        st.warning("Models or data missing. Ensure the training pipeline has executed.")

with tab4:
    st.header("🧠 Feature Importance")
    st.write("Analysis of which features the models rely on most.")
    
    if rf_model is not None:
        X_test_path = os.path.join(MODEL_DIR, "X_test.csv")
        if os.path.exists(X_test_path):
            X_test = pd.read_csv(X_test_path)
            fig_feat, ax_feat = plt.subplots(figsize=(10, 3))
            # Extract best estimator if using GridSearchCV
            if hasattr(rf_model, 'best_estimator_'):
                importances = rf_model.best_estimator_.feature_importances_
            else:
                importances = rf_model.feature_importances_
                
            features = X_test.columns.tolist()
            sns.barplot(x=importances, y=features, ax=ax_feat, palette="viridis", orient='h')
            ax_feat.set_xlabel("Relative Importance in Decision Trees")
            st.pyplot(fig_feat)
        else:
            st.error("Test sets not found.")
            
        st.info("💡 **Interpretation:** The chart above ranks the factors that determine a team's success according to the machine learning model. \n\n- **FIFA Ranking** is overwhelmingly the strongest predictor, meaning sustained historical dominance heavily influences World Cup success.\n- **Average Caps (Experience)** is the second most important factor; teams that have played together frequently handle tournament pressure much better.\n- **Average Age** plays a smaller but vital role; teams typically need a 'Goldilocks' average age around 27—not too young and inexperienced, but not too old and prone to fatigue.")
    else:
        st.warning("Models missing.")

with tab5:
    st.header("⚽ Top 5 World Cup Contenders")
    st.markdown("Based on squad strength, experience, and historical dominance, the **Logistic Regression** model assigns the following win probabilities against a tournament baseline.")
    
    if df_preds is not None:
        top_5 = df_preds.head(5)
        
        # Flag URL mapping
        flag_map = {
            "Spain": "es", "France": "fr", "Argentina": "ar",
            "England": "gb-eng", "Morocco": "ma", "Brazil": "br",
            "Norway": "no", "Germany": "de", "Sweden": "se",
            "Belgium": "be", "Croatia": "hr"
        }
        
        # Display large HTML Cards
        cols = st.columns(5)
        for i, (idx, row) in enumerate(top_5.iterrows()):
            with cols[i]:
                team = row['Team']
                flag_code = flag_map.get(team, "un")
                flag_url = f"https://flagcdn.com/w80/{flag_code}.png"
                
                # Custom HTML Card
                st.markdown(f"""
                <div style="background-color: #1e1e2e; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center; border: 1px solid #333;">
                    <img src="{flag_url}" height="40" style="border-radius: 4px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);">
                    <h5 style="margin: 0; color: #f8f9fa;">Rank #{i+1}: {team}</h5>
                    <h2 style="margin: 10px 0; color: #3b82f6;">{row['Win_Probability']*100:.1f}%</h2>
                    <p style="margin: 0; color: #10b981; font-size: 13px; font-weight: bold;">↑ Rank {int(row['FIFA_Ranking'])} (FIFA)</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        st.markdown("### 🌍 Full Tournament Predictions (48 Teams)")
        st.write("Complete probabilistic standings for all qualified/projected teams.")
        
        # Create a clean display dataframe
        display_df = df_preds.copy()
        display_df.index = display_df.index + 1  # 1-indexed for Rank
        display_df.index.name = "Rank"
        st.dataframe(display_df.style.format({'Win_Probability': '{:.2%}'}), use_container_width=True)
