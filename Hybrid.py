# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

# ✅ Step 2: Load Dataset
train_path = "/content/UNSW_NB15_training-set.csv"
test_path = "/content/UNSW_NB15_testing-set.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Combine train and test datasets for feature selection
df = pd.concat([df_train, df_test], axis=0)

# ✅ Step 3: Data Preprocessing
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# ✅ Fix: Remove "label" if it exists in numerical_cols
numerical_cols = [col for col in numerical_cols if col in df.columns and col != 'label']

# Encode categorical features (protocol, service, state, and attack_cat)
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ✅ Fix: Check if "label" column exists before dropping
columns_to_drop = ['attack_cat']
if 'label' in df.columns:
    columns_to_drop.append('label')  # Drop "label" only if it exists

# Separate features and target
X = df.drop(columns=columns_to_drop)  # Drop attack label (and "label" if present)
y = df['attack_cat']  # Target labels

# ✅ Fix: Apply Normalization only on existing numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# ✅ Step 4: Feature Selection - Compute Information Gain Ratio (IGR)
igr_scores = mutual_info_classif(X, y)
igr_features = pd.Series(igr_scores, index=X.columns).sort_values(ascending=False)

# ✅ Step 5: Feature Selection - Compute Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_scores = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# ✅ Step 6: Hybrid Feature Selection (Intersection of Top Features)
top_igr_features = set(igr_features.head(15).index)  # Top 15 IGR features
top_rf_features = set(rf_scores.head(15).index)  # Top 15 RF features
selected_features = list(top_igr_features & top_rf_features)  # Intersection

print("\n✅ Selected Hybrid Features:", selected_features)

# ✅ Step 7: Train-Test Split Using Selected Features
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, stratify=y, random_state=42)

# ✅ Step 8: Train Base Classifiers (Binary Classifiers for Each Attack Type)
base_classifiers = {}
unique_classes = np.unique(y_train)

for attack in unique_classes:
    y_binary = (y_train == attack).astype(int)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_binary)
    base_classifiers[attack] = clf

# ✅ Step 9: Stacking Ensemble - Generate Meta Features
def stacking_predictions(X, base_classifiers):
    predictions = np.zeros((X.shape[0], len(base_classifiers)))
    for i, (attack, model) in enumerate(base_classifiers.items()):
        predictions[:, i] = model.predict_proba(X)[:, 1]  # Probability of attack
    return predictions

meta_train = stacking_predictions(X_train, base_classifiers)
meta_test = stacking_predictions(X_test, base_classifiers)

# ✅ Step 10: Train Meta-Classifier (Logistic Regression)
meta_clf = LogisticRegression()
meta_clf.fit(meta_train, y_train)

# ✅ Step 11: Model Predictions
y_pred = meta_clf.predict(meta_test)

# ✅ Step 12: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.4f}")
print(f"✅ F1-Score: {f1:.4f}")
print(f"✅ Cohen’s Kappa Score: {kappa:.4f}")
print(f"✅ Confusion Matrix:\n{conf_matrix}")

# ✅ Step 13: Visualizing Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_classes, yticklabels=unique_classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Intrusion Detection")
plt.show()
