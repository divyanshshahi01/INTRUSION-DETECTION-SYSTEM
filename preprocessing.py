# ✅ Step 1: Upload and Load Dataset in Google Colab

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define file paths
train_path = "/content/UNSW_NB15_training-set.csv"
test_path = "/content/UNSW_NB15_testing-set.csv"

# ✅ Step 2: Load Dataset
if os.path.exists(train_path) and os.path.exists(test_path):
    print("\n✅ Files found! Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
else:
    print("\n❌ Error: One or both files not found. Please check the filenames.")

# Combine train and test datasets for analysis
df = pd.concat([df_train, df_test], axis=0)

# ✅ Step 3: Dataset Overview
print("\n📌 Dataset Information:")
print(df.info())

# ✅ Step 4: Check for Missing Values
print("\n📌 Missing Values Count:")
print(df.isnull().sum().sum())  # Total missing values

# ✅ Step 5: Display First 5 Rows
print("\n📌 First 5 Rows:")
print(df.head())

# ✅ Step 6: Check Dataset Shape
print("\n📌 Dataset Shape:", df.shape)

# ✅ Step 7: Identify Numerical and Categorical Columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\n📌 Numerical Features:", numerical_cols)
print("\n📌 Categorical Features:", categorical_cols)

# ✅ Step 8: Analyze Attack Categories
if "attack_cat" in df.columns:
    print("\n📌 Attack Category Distribution:")
    print(df["attack_cat"].value_counts())

    # ✅ Attack Category Bar Plot
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df["attack_cat"], order=df["attack_cat"].value_counts().index, palette="viridis")
    plt.title("Attack Category Distribution")
    plt.xlabel("Count")
    plt.ylabel("Attack Category")
    plt.show()

# ✅ Step 9: Encode Categorical Features (Including Attack Category)
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ✅ Encode Attack Category Separately
if "attack_cat" in df.columns:
    df["attack_cat"] = encoder.fit_transform(df["attack_cat"])

print("\n✅ Categorical Encoding Completed!")

# ✅ Step 10: Correlation Heatmap (Numerical Features vs Attack Category)
if "attack_cat" in df.columns:
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation with Attack Categories")
    plt.show()

# ✅ Step 11: Normalize Numerical Features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\n✅ Data Preprocessing Completed!")
