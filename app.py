from joblib import load
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model
model = load('C:\\Users\\Zakariae\\Desktop\\ProjectML\\best_svm_model.joblib')

# Initialize the StandardScaler (not fitted)
scaler_subset = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            float(request.form['COMPACTNESS']),
            float(request.form['MAX.LENGTH_ASPECT_RATIO']),
            float(request.form['SCALED_VARIANCE_MINOR']),
            float(request.form['MAX.LENGTH_RECTANGULARITY'])
        ]

        # Fit the scaler with the features before transforming
        features_scaled = scaler_subset.fit_transform([features])

        # Make a prediction using the loaded model
        prediction = model.predict(features_scaled)[0]

        # Pass the prediction result to the template
        return render_template('result.html', prediction_result=prediction)

    except Exception as e:
        # Handle errors
        error_message = "An error occurred. Please check the server logs for details."
        return render_template('result.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)