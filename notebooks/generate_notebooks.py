import nbformat as nbf
import os

def create_notebook_01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Task 1: Data Cleaning & Feature Engineering\n\nThis notebook merges historical match data with supplementary team statistics, handles missing values, and engineers features necessary for modeling."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import os

# Load data
matches_df = pd.read_csv('../data/raw/historical_matches.csv')
stats_df = pd.read_csv('../data/raw/supplementary_stats.csv')

print("Matches shape:", matches_df.shape)
print("Stats shape:", stats_df.shape)"""),
        nbf.v4.new_markdown_cell("## Data Cleaning\nRemove duplicates and handle missing values."),
        nbf.v4.new_code_cell("""# Drop duplicates
matches_df = matches_df.drop_duplicates()

# Focus on post-1990 data for relevance
matches_df['date'] = pd.to_datetime(matches_df['date'])
matches_df = matches_df[matches_df['date'].dt.year >= 1990]

# Fill missing scores if any
matches_df['home_score'] = matches_df['home_score'].fillna(0)
matches_df['away_score'] = matches_df['away_score'].fillna(0)
"""),
        nbf.v4.new_markdown_cell("## Feature Engineering\nEngineer goal difference, team win rate, and merge with average age/caps."),
        nbf.v4.new_code_cell("""# Goal difference
matches_df['goal_diff'] = matches_df['home_score'] - matches_df['away_score']

# Determine winner (1 for Home, 0 for Draw, -1 for Away)
matches_df['result'] = np.where(matches_df['goal_diff'] > 0, 1, 
                                np.where(matches_df['goal_diff'] < 0, -1, 0))

# Target variable for classification: Did Home Team Win? (1/0)
matches_df['home_win'] = (matches_df['result'] == 1).astype(int)

# Merge with stats (Simulated: assuming home team stats for simplicity of this baseline)
# In a robust model, we would merge both home and away stats and compute the difference.
final_df = pd.merge(matches_df, stats_df, left_on='home_team', right_on='team', how='inner')
final_df = final_df.drop(columns=['team'])

os.makedirs('../data/processed', exist_ok=True)
final_df.to_csv('../data/processed/final_features.csv', index=False)
print("Saved final_features.csv with shape:", final_df.shape)
""")
    ]
    with open('01_data_cleaning_feature_engineering.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Task 2: Model Building & Training\n\nTrain Logistic Regression and Random Forest models."),
        nbf.v4.new_code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('../data/processed/final_features.csv')

# Features and Target
features = ['average_age', 'average_caps', 'fifa_ranking', 'recent_tournament_score']
features = [f for f in features if f in df.columns]

X = df[features]
y = df['home_win']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling: Standardize features by removing the mean and scaling to unit variance.
# This is crucial for Logistic Regression which is sensitive to feature scales.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
# We use Logistic Regression as a fast, interpretable baseline model.
# Hyperparameters: 'C' controls the inverse of regularization strength. Smaller values specify stronger regularization.
lr_param_grid = {'C': [0.1, 1, 10]}
lr = LogisticRegression(max_iter=1000)
# Use 5-fold cross-validation to robustly estimate model performance and avoid overfitting to a single train-test split.
lr_cv = GridSearchCV(lr, lr_param_grid, cv=5)
lr_cv.fit(X_train_scaled, y_train)

# Model 2: Random Forest
# Random Forest is chosen for its ability to capture non-linear relationships between features (e.g. age vs win probability).
# Hyperparameters: 'n_estimators' is number of trees, 'max_depth' limits tree size to prevent overfitting.
rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
rf = RandomForestClassifier(random_state=42)
# 5-fold CV to ensure the model generalizes well.
rf_cv = GridSearchCV(rf, rf_param_grid, cv=5)
rf_cv.fit(X_train_scaled, y_train)

# Save models and test datasets for Streamlit App (ensuring consistent evaluation)
joblib.dump(lr_cv, 'lr_model.pkl')
joblib.dump(rf_cv, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Models and test sets saved successfully.")
""")
    ]
    with open('02_model_training.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Task 3: Model Evaluation\n\nEvaluate models using accuracy, precision, recall, F1, and ROC-AUC. Generate confusion matrices and ROC curves."),
        nbf.v4.new_code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('../data/processed/final_features.csv')
features = ['average_age', 'average_caps', 'fifa_ranking', 'recent_tournament_score']
features = [f for f in features if f in df.columns]

X = df[features]
y = df['home_win']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = joblib.load('scaler.pkl')
lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')

X_test_scaled = scaler.transform(X_test)

def evaluate_model(name, model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds):.4f}")
    print(f"Recall: {recall_score(y, preds):.4f}")
    print(f"F1 Score: {f1_score(y, preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}\\n")
    
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()
    
    fpr, tpr, _ = roc_curve(y, probs)
    plt.plot(fpr, tpr, label=name)

evaluate_model('Logistic Regression', lr_model, X_test_scaled, y_test)
evaluate_model('Random Forest', rf_model, X_test_scaled, y_test)

plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.show()
""")
    ]
    with open('03_evaluation.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Task 4: Feature Importance\n\nVisualize feature importances for both models."),
        nbf.v4.new_code_cell("""import joblib
import matplotlib.pyplot as plt
import numpy as np

lr_model = joblib.load('lr_model.pkl').best_estimator_
rf_model = joblib.load('rf_model.pkl').best_estimator_

features = ['average_age', 'average_caps', 'fifa_ranking', 'recent_tournament_score']
# For a real pipeline, we'd dynamically load the feature list.

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.barh(features, lr_model.coef_[0])
plt.title('Logistic Regression Coefficients')

plt.subplot(1, 2, 2)
plt.barh(features, rf_model.feature_importances_)
plt.title('Random Forest Feature Importances')

plt.tight_layout()
plt.show()
""")
    ]
    with open('04_feature_importance.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_notebook_01()
    create_notebook_02()
    create_notebook_03()
    create_notebook_04()
    print("Notebooks created.")
