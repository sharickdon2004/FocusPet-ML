from flask import Flask, render_template
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
