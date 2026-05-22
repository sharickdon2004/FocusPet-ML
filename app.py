from pyexpat import model

from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/problematic')
def problematic():
    return render_template('problematic.html')

@app.route('/dataset')
def dataset_explicacion():
    csv_path = os.path.join(os.path.dirname(__file__), 'Dataset_Beta_FocusPet_V2 - Dataset_Beta_FocusPet_V2.csv')
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient='records')
    return render_template('dataset.html', dataset=records)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():

    datos = [[
        int(request.form['edad']),
        float(request.form['horas_pantalla_pre']),
        float(request.form['horas_pantalla_post']),
        int(request.form['sesiones_exitosas']),
        int(request.form['sesiones_fallidas']),
        int(request.form['interacciones_rangel']),
        int(request.form['trivias_ganadas']),
        int(request.form['recompensas_canjeadas'])
    ]]

    resultado = model.predict(datos)

    return render_template(
        'index.html',
        prediccion=resultado[0]
    )

if __name__ == '__main__':
    app.run(debug=True)
    