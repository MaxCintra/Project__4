# Import necessary libraries
from flask import Flask, render_template, request
import joblib  
import numpy as np  

# Create a Flask application
app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('model3_v2.joblib')  # Replace 'your_model_file.pkl' with the actual filename

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle user input and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        # Replace 'feature1', 'feature2', etc., with the actual feature names from your dataset
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        feature5 = float(request.form['feature5'])
        feature6 = float(request.form['feature6'])
        feature7 = float(request.form['feature7'])
        feature8 = float(request.form['feature8'])
        # Repeat this for all the features needed for your prediction

        # Make predictions using the loaded machine learning model
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])  # Create an input array
        prediction = model.predict(input_data)

        # You can format the prediction or use it as needed for display
        # For example, you can convert it to a string and round it to a specific number of decimal places
        prediction_str = f"Predicted Result: {round(prediction[0], 2)}"
        return render_template('result.html', prediction=prediction_str)

if __name__ == '__main__':
    app.run(debug=True)