# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# Load the Data
data_path = "human_psychology_dataset.csv"
df = pd.read_csv(data_path)

# Data Overview
print("Dataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nUnique Values per Column:\n")
print(df.nunique())

print("\nDuplicate Rows:", df.duplicated().sum())

# Handle Missing and Duplicate Data
df = df.drop_duplicates()

# Fill missing values
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['number']):
    df[col].fillna(df[col].mean(), inplace=True)

# Univariate Analysis
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], kde=True, bins=15, color='purple')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Distribution of Optimism Score
plt.figure(figsize=(8, 6))
sns.histplot(df['Optimism Score'], kde=True, bins=10, color='orange')
plt.title('Distribution of Optimism Score')
plt.xlabel('Optimism Score')
plt.ylabel('Frequency')
plt.show()

# Distribution of Anxiety Level
plt.figure(figsize=(8, 6))
sns.countplot(x='Anxiety Level', data=df, palette='Blues')
plt.title('Distribution of Anxiety Levels')
plt.xlabel('Anxiety Level')
plt.ylabel('Count')
plt.show()

# Distribution of Self-Esteem Level
plt.figure(figsize=(8, 6))
sns.countplot(x='Self-Esteem Level', data=df, palette='Set2')
plt.title('Distribution of Self-Esteem Levels')
plt.xlabel('Self-Esteem Level')
plt.ylabel('Count')
plt.show()

# Sleep Hours by Anxiety Level
plt.figure(figsize=(8, 6))
sns.boxplot(x='Anxiety Level', y='Sleep Hours', data=df, palette='coolwarm')
plt.title('Sleep Hours by Anxiety Level')
plt.xlabel('Anxiety Level')
plt.ylabel('Sleep Hours')
plt.show()

#Optimism Score vs. Age
plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Optimism Score', data=df, color='green')
plt.title('Optimism Score vs Age')
plt.xlabel('Age')
plt.ylabel('Optimism Score')
plt.grid(True)
plt.show()

# Personality Type Distribution
plt.figure(figsize=(7, 7))
personality_counts = df['Personality Type'].value_counts()
personality_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#ffcc99'])
plt.title('Personality Type Distribution')
plt.ylabel('')
plt.show()

#Self-Esteem Level Distribution
plt.figure(figsize=(7, 7))
self_esteem_counts = df['Self-Esteem Level'].value_counts()
self_esteem_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#ffcc66', '#66ff66'])
plt.title('Self-Esteem Level Distribution')
plt.ylabel('')
plt.show()


# Scatter Plot Matrix
num_cols = ['Age', 'Sleep Hours', 'Daily Screen Time', 'Optimism Score', 'Memory Recall Score']
sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha':0.5})
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()

# Save Cleaned Data
df.to_csv("cleaned_data.csv", index=False)

