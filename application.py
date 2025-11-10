from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import pickle

# Initialize Flask app
application = Flask(__name__)
app = application

# Load trained Random Forest model
rf_model = pickle.load(open('models/random_forest_new2.pkl', 'rb'))

# User-friendly categorical options for UI
sex_options = [("Male", 1), ("Female", 0)]

cp_options = [
    ("Chest pain during exercise (typical angina)", 0),
    ("Chest discomfort not always related to exercise (atypical angina)", 1),
    ("Chest pain not related to the heart (non-anginal)", 2),
    ("No chest pain but heart issue detected (asymptomatic)", 3)
]

exang_options = [
    ("Chest pain/discomfort during exercise", 1),
    ("No chest pain during exercise", 0)
]

slope_options = [
    ("Better heart response during exercise (upsloping)", 0),
    ("Flat response during exercise (possible issue)", 1),
    ("Worse heart response during exercise (downsloping)", 2)
]

ca_options = [
    ('No major blocked vessels', 0),
    ('One major blocked vessel', 1),
    ('Two major blocked vessels', 2),
    ('Three major blocked vessels', 3),
    ('Four major blocked vessels', 4)
]

thal_options = [
    ("Normal blood flow", 1),
    ("Fixed defect (permanent reduced blood flow)", 2),
    ("Reversible defect (temporary reduced blood flow)", 3)
]

# Landing page
@app.route('/')
def home():
    return render_template('landing.html')

# Prediction page
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

# Prediction route
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    # Numeric inputs
    age = float(request.form.get('age'))
    thalach = float(request.form.get('thalach'))
    trtbps_winsorize = float(request.form.get('trtbps_winsorize'))
    oldpeak_winsorize_sqrt = float(request.form.get('oldpeak_winsorize_sqrt'))

    # Categorical inputs
    sex = int(request.form.get('sex'))
    cp = int(request.form.get('cp'))
    exang = int(request.form.get('exang'))
    slope = int(request.form.get('slope'))
    ca = int(request.form.get('ca'))
    thal = int(request.form.get('thal'))

    # Build DataFrame
    input_df = pd.DataFrame([{
        "age": age,
        "thalach": thalach,
        "trtbps_winsorize": trtbps_winsorize,
        "oldpeak_winsorize_sqrt": oldpeak_winsorize_sqrt,
        "sex": sex,
        "cp": cp,
        "exang": exang,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(
        input_df,
        columns=["sex", "cp", "exang", "slope", "ca", "thal"]
    )

    # Align with model's training columns
    model_columns = rf_model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    result = rf_model.predict(input_encoded)[0]

    # Map result into human-friendly text
    result_text = "⚠️ High risk of heart disease" if result == 1 else "✅ Low risk of heart disease"

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
    app.run(host="0.0.0.0", debug=True)
