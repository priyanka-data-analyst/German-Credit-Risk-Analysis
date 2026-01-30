import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. LOAD DATA
try:
    df = pd.read_csv('german_credit_data.csv')
    print("✅ Data Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: 'german_credit_data.csv' not found. Check the file name.")
    exit()

# ---------------------------------------------------------
# 2. CLEANING (The ETL Part)
# ---------------------------------------------------------
# Drop the useless index column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Fill missing values with 'unknown' (Business Logic)
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

print("🧹 Data Cleaned. Missing values handled.")

# ---------------------------------------------------------
# 2.5 SAVE THE CLEANED FILE (This is what you asked for)
# ---------------------------------------------------------
df.to_csv('cleaned_german_credit_data.csv', index=False)
print("💾 SUCCESS: 'cleaned_german_credit_data.csv' has been created in your folder!")

# ---------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------
print("📊 Generating Graph... Check your taskbar.")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Risk', y='Credit amount', data=df)
plt.title("Credit Amount vs Risk")
plt.show()
# NOTE: You must CLOSE the graph window for the script to continue!

# ---------------------------------------------------------
# 4. PREPROCESSING FOR AI (Encoding)
# ---------------------------------------------------------
# Convert 'Risk' to numbers (bad=1, good=0)
le = LabelEncoder()
df['Risk'] = le.fit_transform(df['Risk'])

# Convert categories to numbers (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# 5. TRAIN MODEL
X = df_encoded.drop('Risk', axis=1)  # Features
y = df_encoded['Risk']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. RESULTS
predictions = model.predict(X_test)
print("\n" + "=" * 30)
print("🎯 MODEL ACCURACY REPORT")
print("=" * 30)
print(df.columns)
print(classification_report(y_test, predictions))
