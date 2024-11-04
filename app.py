from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import os
import requests

app = Flask(__name__)
CORS(app)

# URL model dari Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CswCfFUwgZwqyuJzd6vYnjjj6g5a_NCD"

def download_model():
    if not os.path.exists('model_sign_train.h5'):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open('model_sign_train.h5', 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")

# Download dan load model saat startup
try:
    print("Initializing model...")
    download_model()
    model = tf.keras.models.load_model('model_sign_train.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")

def preprocess_image(image):
    # Resize gambar ke ukuran 28x28 sesuai dengan training
    expected_size = (28, 28)
    image = image.resize(expected_size)
    
    # Convert ke grayscale karena model dilatih dengan gambar grayscale
    image = image.convert('L')
    
    # Konversi ke array dan normalisasi
    image = np.array(image) / 255.0
    
    # Reshape sesuai input shape model (None, 28, 28, 1)
    image = image.reshape(1, 28, 28, 1)
    
    # Convert ke float32
    image = image.astype('float32')
    
    return image

# Dictionary untuk mapping label
label_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F", 
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R", 
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "X",
    23: "Y"
}

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Sign Language Recognition API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diberikan'}), 400
    
    file = request.files['file']
    
    try:
        # Baca gambar
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess gambar
        processed_image = preprocess_image(image)
        
        # Debug - print shape
        print(f"Processed image shape: {processed_image.shape}")
        
        # Jalankan prediksi
        prediction = model.predict(processed_image)
        
        # Ambil index kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(prediction[0])
        
        # Ambil confidence score
        confidence = float(prediction[0][predicted_class])
        
        # Get label dari dictionary 
        predicted_label = label_dict.get(predicted_class, "Tidak Dikenal")
        
        # Format response
        result = {
            'prediksi': predicted_label,
            'confidence': f'{confidence:.2%}',
            'index_kelas': int(predicted_class)
        }
        
        print(f"Prediction successful: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Error selama prediksi',
            'pesan': str(e)
        }), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'pesan': 'Flask API sedang berjalan!'})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Endpoint tidak ditemukan',
        'status': 'error'
    }), 404

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        'error': str(e),
        'status': 'error'
    }), 500

if __name__ == '__main__':
    print("Starting application...")
    try:
        # Print model summary untuk debugging
        model.summary()
        print(f"Expected input shape: {model.input_shape}")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        
    # Gunakan PORT dari environment variable untuk Heroku
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)