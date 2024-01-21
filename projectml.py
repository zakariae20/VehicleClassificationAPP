from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("C:\\Users\\Zakariae\\Downloads\\vehicle.csv")

# Select the features you want to use for training and prediction
selected_features = ['COMPACTNESS', 'MAX.LENGTH_ASPECT_RATIO', 'SCALED_VARIANCE_MINOR', 'MAX.LENGTH_RECTANGULARITY']

# Encode the target variable
label_encoder = LabelEncoder()
data['Class'] = label_encoder.fit_transform(data['Class'])

# Prepare the data with only the selected features
Data_subset = data[selected_features]
target_subset = data['Class']

# Split the data into features (X) and target (y)
X_subset = Data_subset
y_subset = target_subset

# Initialize and fit the StandardScaler
scaler_subset = StandardScaler()
X_scaled_subset = scaler_subset.fit_transform(X_subset)

# Split the scaled data into training and testing sets
X_train_scaled_subset, X_test_scaled_subset, y_train_subset, y_test_subset = train_test_split(X_scaled_subset, y_subset, test_size=0.33)

# Initialize the Support Vector Machine (SVM) classifier with RBF kernel
svm_clf = SVC(kernel='rbf')

# Train the SVM classifier
svm_clf.fit(X_train_scaled_subset, y_train_subset)

# Make predictions on the test set
predictions_svm = svm_clf.predict(X_test_scaled_subset)

# Evaluate the accuracy
accuracy_svm = accuracy_score(y_test_subset, predictions_svm)
print("Accuracy with SVM (RBF Kernel):", accuracy_svm)

# Save the SVM model
dump(svm_clf, 'best_svm_rbf_model.joblib')
