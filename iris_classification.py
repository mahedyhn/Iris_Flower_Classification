import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset 
# Ensure the filename matches your file in the sidebar exactly!
df = pd.read_csv('iris.csv')

# 2. Data Cleaning
# Drop 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# 3. Splitting Features and Target
X = df.drop('Species', axis=1)
y = df['Species']

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Building
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualization (Confusion Matrix)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Iris Flower Classification - Confusion Matrix')
plt.show()

# 8. Testing with Custom Data (Removing the Warning)
print("\n--- Testing with Custom Data ---")
sample_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                           columns=X.columns) # This ensures feature names match

prediction = model.predict(sample_data)
print(f"Input Data: [[5.1, 3.5, 1.4, 0.2]]")
print(f"Predicted Species: {prediction[0]}")