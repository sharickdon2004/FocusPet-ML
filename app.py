from flask import Flask, render_template, request
import pandas as pd
import os
from decisiontree import model
from randomforest import model as randomforest_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/problematic')
def problematic():
    return render_template('problematic.html')

@app.route('/dataset')
def dataset_explicacion():
    csv_path = os.path.join(os.path.dirname(__file__), 'DataSet_FocusPet.csv')
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
                float(request.form['age']),
                float(request.form['screen_time_pre']),
                float(request.form['screen_time_post']),
                float(request.form['successful_sessions']),
                float(request.form['failed_sessions']),
                float(request.form['rangel_interactions']),
                float(request.form['trivias_won']),
                float(request.form['rewards_redeemed'])
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



@app.route('/randomforest', methods=['GET', 'POST'])
def randomforest():

    resultado = None

    if request.method == 'POST':
        resultado = randomforest_model.procesar(request.form)

    return render_template(
        'randomforest.html',
        resultado=resultado
    )



if __name__ == '__main__':
    app.run(debug=True)