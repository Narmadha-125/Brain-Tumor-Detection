from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import random
from playsound import playsound
from threading import Thread
from fpdf import FPDF
import datetime

app = Flask(__name__)
model = joblib.load('model.pkl')

# Healthy quotes
quotes = [
    "A healthy outside starts from the inside.",
    "Your body hears everything your mind says — stay positive.",
    "Eat well, move more, feel strong.",
    "Healing begins with awareness.",
    "Health is the crown on the well person’s head.",
     "Health is wealth. Protect your mind.",
    "Early detection saves lives.",
    "A healthy outside starts from the inside.",
    "Wellness begins with awareness."
]

# Doctor suggestions
doctors = [
    {"name": "Dr. Aarthi Ramesh", "hospital": "Apollo Hospitals, Chennai"},
    {"name": "Dr. Vikram Shetty", "hospital": "Manipal Hospitals, Bangalore"},
    {"name": "Dr. Rajesh Iyer", "hospital": "AIIMS, Delhi"},
    {"name": "Anita Raj", "hospital": "Apollo Hospitals", "location": "Chennai"},
    {"name": "Ravi Kumar", "hospital": "AIIMS", "location": "New Delhi"},
    {"name": "Sahana N", "hospital": "CMC", "location": "Vellore"},
    {"name": "Dev Sharma", "hospital": "Fortis", "location": "Mumbai"}
]



# Route for index.html
@app.route('/')
def index():
    quote = random.choice(quotes)
    return render_template('index.html', quote=quote)

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Limit to 2MB

    # Process image
    img = Image.open(file).convert('L').resize((100, 100))
    img_array = np.array(img).flatten().reshape(1, -1)
    pred = model.predict(img_array)[0]

    if pred == 1:
        result = "Tumor Detected"
        suggestions = {
        "do": ["Eat leafy greens", "Exercise regularly", "Drink plenty of water"],
        "dont": ["Avoid processed meats", "Limit sugar", "Do not smoke or drink"]
    }
        doctor = random.choice(doctors)
        
    else:
        result = "No Tumor Detected"
        suggestions = None
        doctor = None
        
    return render_template(
    'result.html',
    result=result,
    suggestions=suggestions,
    doctor=doctor,
    quote= random.choice(quotes)
)



# Generate downloadable PDF report
@app.route('/download_report', methods=['POST'])
def download_report():
    result = request.form.get('result')
    doctor = request.form.get('doctor')
    date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Brain Tumor Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {date}", ln=True)
    pdf.cell(200, 10, txt=f"Result: {result}", ln=True)

    if doctor:
        pdf.cell(200, 10, txt=f"Recommended Doctor: {doctor}", ln=True)

    filename = "report.pdf"
    pdf.output(filename)
    return send_file(filename, as_attachment=True)

# Flask server start
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
