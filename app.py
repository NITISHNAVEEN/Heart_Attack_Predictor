from flask import Flask, request, render_template
import numpy as np
import pickle

# Initialize Flask app
application = Flask(__name__)
app = application

# Load trained Random Forest model
try:
    rf_model = pickle.load(open('models/random_forest_new2.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    rf_model = None

# Keep your existing options definitions
sex_options = [("Male", 1), ("Female", 0)]
# ... other options remain the same ...

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict')
def index():
    return render_template(
        'index.html',
        sex_options=sex_options,
        cp_options=cp_options,
        exang_options=exang_options,
        slope_options=slope_options,
        ca_options=ca_options,
        thal_options=thal_options,
        result=None
    )

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Create feature array
        features = np.zeros(32)  # Adjust size based on your model's features
        
        # Set numeric features
        features[0] = float(request.form.get('age'))
        features[1] = float(request.form.get('thalach'))
        features[2] = float(request.form.get('trtbps_winsorize'))
        features[3] = float(request.form.get('oldpeak_winsorize_sqrt'))
        
        # Get categorical features
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        exang = int(request.form.get('exang'))
        slope = int(request.form.get('slope'))
        ca = int(request.form.get('ca'))
        thal = int(request.form.get('thal'))
        
        # Set one-hot encoded features
        features[4 + sex] = 1  # sex
        features[6 + cp] = 1   # cp
        features[10 + exang] = 1  # exang
        features[12 + slope] = 1  # slope
        features[15 + ca] = 1    # ca
        features[20 + thal] = 1  # thal
        
        # Make prediction
        input_features = features.reshape(1, -1)
        result = rf_model.predict(input_features)[0]
        
        result_text = "⚠️ High risk of heart disease" if result == 1 else "✅ Low risk of heart disease"
        
    except Exception as e:
        print(f"Prediction error: {e}")
        result_text = "Error processing request"

    return render_template(
        'index.html',
        sex_options=sex_options,
        cp_options=cp_options,
        exang_options=exang_options,
        slope_options=slope_options,
        ca_options=ca_options,
        thal_options=thal_options,
        result=result_text
    )

if __name__ == "__main__":
    app.run(debug=True)