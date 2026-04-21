from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import json

app = Flask(__name__)

model = None
model_stats = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global model, model_stats
    data = request.get_json()
    
    x_raw = data.get('x_values', [])
    y_raw = data.get('y_values', [])
    
    if len(x_raw) < 2 or len(y_raw) < 2:
        return jsonify({'error': 'Please provide at least 2 data points.'}), 400
    
    if len(x_raw) != len(y_raw):
        return jsonify({'error': 'X and Y must have the same number of values.'}), 400

    try:
        X = np.array(x_raw, dtype=float).reshape(-1, 1)
        y = np.array(y_raw, dtype=float)
    except ValueError:
        return jsonify({'error': 'All values must be valid numbers.'}), 400

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    model_stats = {
        'slope': round(float(model.coef_[0]), 6),
        'intercept': round(float(model.intercept_), 6),
        'r2': round(float(r2), 6),
        'rmse': round(float(rmse), 6),
        'n_samples': len(x_raw)
    }

    # Generate line points for chart
    x_min, x_max = float(np.min(X)), float(np.max(X))
    x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    return jsonify({
        'stats': model_stats,
        'scatter': {'x': x_raw, 'y': y_raw},
        'line': {'x': x_line.flatten().tolist(), 'y': y_line.tolist()}
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400

    data = request.get_json()
    try:
        x_val = float(data.get('x_input'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid input. Please enter a numeric value.'}), 400

    prediction = model.predict(np.array([[x_val]]))[0]
    return jsonify({
        'x': x_val,
        'prediction': round(float(prediction), 6)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 6969)
