from flask import Flask, render_template, request
import pandas as pd
import os
from decisiontree import model

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

@app.route('/decisiontree', methods=['GET', 'POST'])
def decisiontree():
    prediction = None
    title = None
    description = None
    motivation = None
    error = None

    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            datos = [
                float(request.form['edad']),
                float(request.form['horas_pantalla_pre']),
                float(request.form['horas_pantalla_post']),
                float(request.form['sesiones_exitosas']),
                float(request.form['sesiones_fallidas']),
                float(request.form['interacciones_rangel']),
                float(request.form['trivias_ganadas']),
                float(request.form['recompensas_canjeadas'])
            ]
            
            # Realizar predicción
            prediction = model.predict(datos)
            
            # Obtener mensajes personalizados
            messages = model.get_messages(prediction)
            title = messages['title']
            description = messages['description']
            motivation = messages['motivation']
            
            print(f" Predicción: {prediction} - {title}")
            
        except Exception as e:
            error = f"Error en la predicción: {str(e)}"
            print(f" {error}")

    return render_template('decisiontree.html', 
                         prediction=prediction,
                         title=title,
                         description=description,
                         motivation=motivation,
                         error=error,
                         accuracy=model.get_accuracy())

if __name__ == '__main__':
    app.run(debug=True)