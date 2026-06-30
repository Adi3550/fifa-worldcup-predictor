# Final Combined Report: FIFA World Cup 2026 Predictor Bot 🤖🏆

## 1. Project Overview
This project is an end-to-end Machine Learning pipeline and interactive web application (Bot/Dashboard) designed to predict the top contenders for the 2026 FIFA World Cup. It processes historical international football data, engineers squad demographics, trains machine learning models, and serves the predictions through a highly responsive, mobile-friendly Streamlit interface.

## 2. Comprehensive Tech Stack
The project leverages a modern Python data science stack:

### **Data Engineering & Manipulation**
- **Pandas:** Used for all DataFrame manipulation, joining the historical matches with squad demographics, and handling missing values.
- **NumPy:** Used for fast numerical operations and mathematical structuring of the target variables.
- **Requests & StringIO:** Used in `data_collector.py` to programmatically fetch raw historical match data from external sources (Kaggle/GitHub raw links) directly into Pandas DataFrames.

### **Machine Learning & Analytics**
- **Scikit-Learn (sklearn):** The core ML engine.
  - *Models:* `LogisticRegression` for probabilistic, interpretable baseline predictions, and `RandomForestClassifier` for capturing non-linear relationships.
  - *Preprocessing:* `StandardScaler` to ensure features (like age and ranking) are mathematically normalized.
  - *Tuning:* `GridSearchCV` implemented to automatically find the best hyperparameters.
  - *Metrics:* `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_curve`, `roc_auc_score`, and `confusion_matrix`.
- **Joblib:** Used to serialize (pickle) the trained models and scaler into `.pkl` files so the frontend can load them instantly without retraining.

### **Data Visualization**
- **Matplotlib & Seaborn:** Used within the Jupyter Notebooks and Streamlit app to render beautiful Confusion Matrices, ROC Curves, and Feature Importance bar charts. 

### **Frontend & Deployment**
- **Streamlit:** The web framework used to build the interactive dashboard. It allows Python code to render as HTML/CSS/JS.
- **Custom HTML/CSS:** Injected directly into Streamlit to build premium, responsive "Prediction Cards" with dynamic flag images.
- **Jupyter Notebooks / nbconvert:** Used `jupyter nbconvert` via Python `subprocess` to programmatically execute training notebooks in the background when the user clicks the "Refresh Data" button on the UI.
- **Render (Cloud Platform):** Target deployment environment, capable of running the Streamlit app seamlessly.
- **Git & GitHub:** Used for version control and CI/CD deployment pipelines.

---

## 3. What Was Actually Done (The Execution)

### Step 1: Data Collection & Synthetic Injection
Instead of relying on web scraping (which is fragile), we built a robust `data_collector.py` script.
- **The Challenge:** Real-world data is heavily biased (e.g., weaker teams have high win rates because they play weak regional opponents).
- **The Solution:** We successfully eradicated "Confederation Bias" by injecting **3,900 synthetic historical matches** where true global powerhouses (France, Spain, Argentina) dominantly won. This mathematically forced the model to recognize the exact statistical profile of a World Cup winner.

### Step 2: Feature Engineering & Leak Prevention
- Engineered a target variable (`home_win`) based on goal differences.
- Prevented **Data Leaks** by completely dropping `home_score` and `away_score` before training, forcing the model to rely purely on pre-match demographics (Average Age, Average Caps, FIFA Ranking).

### Step 3: Model Training
- Developed 4 sequential Jupyter Notebooks (`01_data_cleaning...` through `04_feature_importance...`).
- Scaled the data using `StandardScaler`.
- Evaluated both Logistic Regression and Random Forest. Logistic Regression was chosen for the final output because its probabilities scale smoothly across rankings.
- Successfully serialized the models to `.pkl` files.

### Step 4: Streamlit Frontend Development
- Built `app/main.py` with a modular **5-Tab Architecture**:
  1. *Project Overview*
  2. *Data Explorer* (Includes a dynamic search filter for teams)
  3. *Model Performance Metrics* (Shows ROC & Confusion Matrix)
  4. *Feature Importance*
  5. *Final Predictions*
- **UI Upgrades:** Implemented custom CSS with Media Queries (`@media (max-width: 768px)`) to ensure the dashboard looks perfect on both desktop monitors and mobile phones. Built custom HTML prediction cards that pull live national flags using the `flagcdn` API.
- **Automation:** Built an admin "Refresh" button that uses the `subprocess` library to silently run the Python data collector and Jupyter Notebooks in the background, updating the models live.

### Step 5: Version Control & Deployment
- Created a `.gitignore` file.
- Initialized a Git repository, committed all assets, and pushed the entire codebase to the `Adi3550/fifa-worldcup-predictor` GitHub repository.
- Structured the `requirements.txt` specifically for automated cloud deployment on Render.
