# Import necessary libraries
from flask import Flask, render_template, request
import joblib  
import numpy as np  

# Create a Flask application
app = Flask(__name__)

# Load your trained machine learning model
model1 = joblib.load('probowl_prediction.joblib')  
model2 = joblib.load('allpro_prediction.joblib')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle user input and make predictions
@app.route('/predict', methods=['GET', 'POST'])
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
        feature9 = float(request.form['feature9'])
        feature10 = float(request.form['feature10'])
        feature11 = float(request.form['feature11'])
        feature12 = float(request.form['feature12'])
        # Repeat this for all the features needed for your prediction

        # Make predictions using the loaded machine learning model
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12]])  # Create an input array
        prediction1 = model1.predict(input_data)
        prediction2 = model2.predict(input_data)
        # You can format the prediction or use it as needed for display
        # For example, you can convert it to a string and round it to a specific number of decimal places
        prediction_str1 = f"Predicted Result: {round(prediction1[0], 2)}"
        prediction_str2 = f"Predicted Result: {round(prediction2[0], 2)}"
        return render_template('result.html', prediction1=prediction_str1, prediction2=prediction_str2)
    
if __name__ == '__main__':
    app.run(debug=True)