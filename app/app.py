import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from flask import Flask, request, jsonify, render_template  # Importamos render_template
from PIL import Image
import io
import numpy as np

# Crear la aplicación Flask
app = Flask(__name__)

# Definir el modelo
def get_model():
    net = nn.Sequential()
    net.add(
        nn.Dense(256, activation="relu"),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
    return net

# Cargar el modelo preentrenado
def load_model():
    net = get_model()
    net.load_parameters('model/cifar10_model.params', ctx=mx.cpu())  # Ajusta el contexto según sea necesario
    return net

# Ruta de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if 'img' in request.files:
            # imagen enviada
            img = Image.open(io.BytesIO(request.files['img'].read()))
            
            # Redimensionar la imagen a 32x32 (o el tamaño adecuado)
            img = img.resize((32, 32)) 
            
            # Convertir la imagen a un arreglo de NumPy
            img = np.array(img)
            
            # Asegurarse de que la imagen tenga 3 canales (RGB)
            if img.shape[2] == 4: 
                img = img[:, :, :3]
            
            # Convertir a un tensor MXNet
            img = nd.array(img).astype(np.float32) 
            img = img.transpose((2, 0, 1))  
            img = img.expand_dims(axis=0)  
            
            img = img / 255.0  
            
            # Cargar el modelo
            net = load_model()

            # Realizar la predicción
            pred = net(img)
            predicted_class = int(nd.argmax(pred, axis=1).asscalar())  # Obtener la clase predicha
            
            # Nombres de las clases 
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            prediction = class_names[predicted_class]

            # Devolver la predicción en formato JSON
            return jsonify({'prediction': prediction})

    return jsonify({'error': 'No image provided'}), 400

# Ruta para mostrar el formulario HTML
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")