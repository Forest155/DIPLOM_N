import pickle
import numpy as np
from flask import Flask, request, jsonify

print("🚀 Запуск приложения")

# Загрузка модели и списка признаков
model = pickle.load(open('model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return '<h2>API работает. Отправьте POST-запрос на /predict</h2>'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    try:
        X_input = np.array([data[feat] for feat in features]).reshape(1, -1)
    except KeyError as e:
        return jsonify({'error': f'Признак {e} не найден во входных данных'}), 400

    prediction = model.predict(X_input)
    return jsonify({'predicted_price': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)