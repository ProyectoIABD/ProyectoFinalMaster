from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import joblib

import pandas as pd

DATA = pd.read_csv('../../data/processed/full_data.csv')
PATH_MODELS = '../modeling/trained'
PATH_MODEL_COLOR = f'{PATH_MODELS}/modelo_color.pt'
PATH_MODEL_TEXT = f'{PATH_MODELS}/modelo_texto.pkl'
PATH_MODEL_PRICE = f'{PATH_MODELS}/modelo_precio.pkl'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Cargar modelo de ML entrenado previamente (por ejemplo, con scikit-learn)
model = joblib.load(PATH_MODEL_TEXT)  # Asegúrate de tener este archivo en tu proyecto

@app.route('/')
def index():
    car_types = sorted(DATA['car_type'].unique())
    return render_template('index.html', car_types=car_types)

@app.route('/predict', methods=['POST'])
def predict():
    car_type = request.form.get('car_type')
    year = request.form.get('year')
    km = request.form.get('km')
    description = request.form.get('description')
    image = request.files.get('image')

    # Validación
    if not (car_type and year and km and description and image):
        return "Todos los campos son obligatorios", 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Procesamiento para el modelo (ejemplo)
    description_length = len(description.split())
    input_features = [[car_type, int(year), int(km), description_length]]

    # Predicción ejemplo
    predicted_price = 12300
    #model.predict(input_features)

    # Pasamos toda la info a la plantilla
    return render_template(
        'result.html',
        price=predicted_price,
        car_type=car_type,
        year=year,
        km=km,
        description=description,
        image_path=filepath
    )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)