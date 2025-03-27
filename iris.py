import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
file_path = "IRIS.csv"
df = pd.read_csv(file_path)

# Check for missing values
print(df.isnull().sum())

# Encode categorical labels
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Split Features and Target
X = df.drop(columns=['species'])  # Features (Petal/Sepal measurements)
y = df['species']  # Target (Species)

# Split Data into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature Importance Analysis
feature_importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]

print("\nFeature Importances:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {feature_importances[i]:.4f}")

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Perform Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation Accuracy:", np.mean(cv_scores))

# Confusion Matrix Visualization
plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
