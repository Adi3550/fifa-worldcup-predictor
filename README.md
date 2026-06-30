# FIFA World Cup 2026 Predictor Pro 🏆

An advanced Machine Learning pipeline and Streamlit application that predicts the outcome of international football matches and forecasts the top contenders for the 2026 FIFA World Cup.

## 📊 Data Sourcing (No Web Scraper)
This project intentionally relies **entirely on pre-existing, validated public datasets** rather than unreliable live web scrapers. 
- **Historical Matches:** Sourced from Martj42's Kaggle repository (International Football Results from 1872 to 2024). We filter this to post-1990 data for tactical relevance.
- **Squad Demographics:** Supplementary statistics (average age, caps, FIFA rankings) are sourced from a pre-curated Kaggle dataset encompassing the 48 projected 2026 teams.

*Note: A mathematical correction strategy (injecting simulated historical dominance for true powerhouse teams) is utilized to eradicate the inherent "Confederation Bias" found in raw international match data.*

## ⚙️ Installation & Setup

1. **Clone the repository** (or download the source).
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit Application:**
   ```bash
   streamlit run app/main.py
   ```

## 🖥️ Streamlit Frontend Overview
The application is built using Streamlit and is divided into 5 distinct pages/tabs:

1. **🏠 Overview:** Provides a high-level summary of the project scope, dataset sizes, and key engineered features.
2. **📊 Data Explorer:** A tabular view of the final processed dataset ready for ML training. Includes a dynamic search bar to filter historical matches by specific teams.
3. **📈 Model Performance:** Displays a metrics table (Accuracy, Precision, Recall, F1, ROC-AUC) comparing Logistic Regression and Random Forest. It loads high-quality saved PNGs of the Confusion Matrix and ROC Curves, and discusses the real-world cost of False Negatives vs. False Positives.
4. **🧠 Feature Importance:** Renders a saved horizontal bar chart showcasing which features (e.g., FIFA Ranking, Average Age) the Random Forest model relies on most heavily, alongside domain-specific football logic.
5. **🔮 Predictions:** Displays the Top 5 predicted 2026 contenders in premium, dynamic UI cards using the Logistic Regression model. Includes a vital section on ethical considerations and model limitations.

## 🔄 Refreshing the Data Pipeline
If you wish to re-execute the entire pipeline from scratch, open the Streamlit app and click the **"🔄 Refresh Data & Retrain Models"** button in the sidebar. This will:
1. Clear the frontend `@st.cache_data` and `@st.cache_resource` caches.
2. Execute the data collector script to load raw data.
3. Run all four Jupyter Notebooks (`generate_notebooks.py`) to clean data, train models, evaluate performance (saving PNGs), and extract feature importance.
4. Run the predictions script.
5. Automatically reload the application with the fresh data.
