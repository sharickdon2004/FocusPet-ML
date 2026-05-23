import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

def train_and_save_model():
    # 1. Cargar los datos intentando detectar el formato automáticamente
    try:
        # Intento 1: Separado por punto y coma (;)
        df = pd.read_csv('DataSet_FocusPet.csv', sep=';', decimal=',')
        if 'age' not in df.columns: # Si no separó bien las columnas, forzamos el error
            raise ValueError
    except:
        # Intento 2: Separado por comas (formato CSV estándar)
        df = pd.read_csv('DataSet_FocusPet.csv', sep=',', decimal='.')

    # Verificamos si logramos leer las columnas correctamente
    if 'age' not in df.columns:
        print("¡Error crítico! Las columnas detectadas son:", df.columns)
        print("Asegúrate de que el archivo DataSet_FocusPet.csv tenga el encabezado correcto.")
        return

    # 2. Definir las variables predictoras (X) y la variable objetivo a predecir (y)
    features = [
        'age', 'screen_time_pre', 'screen_time_post', 
        'successful_sessions', 'failed_sessions', 
        'rangel_interactions', 'trivias_won', 'rewards_redeemed'
    ]
    X = df[features]
    y = df['beta_rating']

    # 3. Dividir los datos: 80% para entrenamiento, 20% para prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Crear y entrenar el modelo de Regresión Lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Evaluar la precisión del modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Entrenamiento completado.")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Coeficiente de Determinación (R2): {r2:.4f}")

    # 6. Guardar el modelo en formato .pkl
    joblib.dump(model, 'linear_regression_model.pkl')
    print("Modelo guardado exitosamente como 'linear_regression_model.pkl'")

    # 7. Generar y guardar la gráfica
    os.makedirs('static/images', exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='#007BFF', alpha=0.7, label='Predicciones vs Reales')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2, label='Ajuste Perfecto')
    plt.xlabel('Calificación Real (beta_rating)')
    plt.ylabel('Calificación Predicha')
    plt.title('Regresión Lineal: Valores Reales vs Predichos')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig('static/images/lr_results.png', bbox_inches='tight')
    plt.close()
    print("Gráfica guardada en 'static/images/lr_results.png'")

if __name__ == "__main__":
    train_and_save_model()