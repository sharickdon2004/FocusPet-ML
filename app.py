from flask import Flask, render_template, request
import pandas as pd
import os
import joblib
from decisiontree import model
from randomforest import model as randomforest_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/problematic")
def problematic():
    return render_template("problematic.html")


@app.route("/dataset")
def dataset_explicacion():
    csv_path = os.path.join(os.path.dirname(__file__), "DataSet_FocusPet.csv")
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    return render_template("dataset.html", dataset=records)


@app.route("/decisiontree", methods=["GET", "POST"])
def decisiontree():
    prediction = None
    title = None
    description = None
    motivation = None
    error = None

    if request.method == "POST":
        try:
            # Obtener datos del formulario
            datos = [
                float(request.form["age"]),
                float(request.form["screen_time_pre"]),
                float(request.form["screen_time_post"]),
                float(request.form["successful_sessions"]),
                float(request.form["failed_sessions"]),
                float(request.form["rangel_interactions"]),
                float(request.form["trivias_won"]),
                float(request.form["rewards_redeemed"]),
            ]

            # Realizar predicción
            prediction = model.predict(datos)

            # Obtener mensajes personalizados
            messages = model.get_messages(prediction)
            title = messages["title"]
            description = messages["description"]
            motivation = messages["motivation"]

            print(f" Predicción: {prediction} - {title}")

        except Exception as e:
            error = f"Error en la predicción: {str(e)}"
            print(f" {error}")

    return render_template(
        "decisiontree.html",
        prediction=prediction,
        title=title,
        description=description,
        motivation=motivation,
        error=error,
        accuracy=model.get_accuracy(),
    )


@app.route("/randomforest", methods=["GET", "POST"])
def randomforest():

    resultado = None

    if request.method == "POST":
        resultado = randomforest_model.procesar(request.form)

    return render_template("randomforest.html", resultado=resultado)


@app.route("/linearregression", methods=["GET", "POST"])
def linearregression():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            datos = [
                float(request.form["age"]),
                float(request.form["screen_time_pre"]),
                float(request.form["screen_time_post"]),
                float(request.form["successful_sessions"]),
                float(request.form["failed_sessions"]),
                float(request.form["rangel_interactions"]),
                float(request.form["trivias_won"]),
                float(request.form["rewards_redeemed"]),
            ]

            model_path = os.path.join(os.path.dirname(__file__), "linear_regression_model.pkl")
            lr_model = joblib.load(model_path)         
            pred = lr_model.predict([datos])[0]
            prediction = round(max(1.0, min(5.0, pred)), 2)

            print(f"Predicción de Regresión Lineal exitosa: {prediction}")

        except Exception as e:
            error = f"Error en la predicción: {str(e)}"
            print(error)

    return render_template("regression.html", prediction=prediction, error=error)


@app.route("/business-understanding")
def business_understanding():
    return render_template("business_understanding.html")

@app.route("/data-understanding")
def data_understanding():
    return render_template("data_understanding.html")

@app.route("/data-engineering")
def data_engineering():
    return render_template("data_engineering.html")

@app.route("/model-engineering")
def model_engineering():
    return render_template("model_engineering.html")

@app.route("/model-evaluation")
def model_evaluation():
    return render_template("model_evaluation.html")


if __name__ == "__main__":
    app.run(debug=True)