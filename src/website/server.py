from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

import joblib
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image
import pandas as pd

# === Configuración y carga de modelos ===

DATA = pd.read_csv('../../data/processed/full_data.csv')
PATH_MODELS = '../modeling/trained'
PATH_MODEL_COLOR = f'{PATH_MODELS}/modelo_color.pt'
PATH_MODEL_TEXT = f'{PATH_MODELS}/modelo_texto.pkl'
PATH_MODEL_PRICE = f'{PATH_MODELS}/xgb.pkl'
PATH_ENCODER_TYPE = f'{PATH_MODELS}/le_car_type.pkl'

IMG_SIZE = 128

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# CNN
class CarCNN(nn.Module):
    def __init__(self, num_classes):
        super(CarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Modelos y utilidades
price_model = joblib.load(PATH_MODEL_PRICE)
color_model = CarCNN(num_classes=15)
color_model.load_state_dict(torch.load(PATH_MODEL_COLOR, map_location=torch.device('cpu')))
text_model = joblib.load(PATH_MODEL_TEXT)
le_car_type = joblib.load(PATH_ENCODER_TYPE)
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# === Funciones auxiliares ===

def predict_color(image_path, model):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
        return torch.argmax(pred, dim=1).item()

def predict_estado(description, model, transformer):
    embedding = transformer.encode([description])
    return model.predict(embedding)[0]

def encode_estado(estado_str):
    return {
        'estado__bueno': int(estado_str == 'bueno'),
        'estado__malo': int(estado_str == 'malo'),
        'estado__neutral': int(estado_str == 'neutral')
    }

def encode_car_type(car_type, encoder):
    return encoder.transform([car_type])[0]

def construir_input(year, km, color_encoded, car_type_encoded, estado_dict):
    return [[
        int(year),
        float(km),
        color_encoded,
        car_type_encoded,
        estado_dict['estado__bueno'],
        estado_dict['estado__malo'],
        estado_dict['estado__neutral']
    ]]

# === Rutas de la aplicación ===

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

    if not (car_type and year and km and description and image):
        return "Todos los campos son obligatorios", 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Predicciones
    color = predict_color(filepath, color_model)
    estado = predict_estado(description, text_model, sentence_transformer)
    estado_dict = encode_estado(estado)
    car_type_encoded = encode_car_type(car_type, le_car_type)

    input_features = construir_input(year, km, color, car_type_encoded, estado_dict)
    predicted_price = price_model.predict(input_features)[0]

    return render_template(
        'result.html',
        price=round(predicted_price, 2),
        car_type=car_type,
        year=year,
        km=km,
        description=description,
        estado=estado,
        color=color,
        image_path=filepath
    )

# === Iniciar la app ===

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)