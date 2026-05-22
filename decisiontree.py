import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset
df = pd.read_csv('dataset.csv')

# Variables de entrada
X = df[[
    'edad',
    'horas_pantalla_pre',
    'horas_pantalla_post',
    'sesiones_exitosas',
    'sesiones_fallidas',
    'interacciones_rangel',
    'trivias_ganadas',
    'recompensas_canjeadas'
]]

# Variable objetivo
Y = df['calificacion_beta']

# Dividir datos
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Crear modelo
modelo = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

# Entrenar
modelo.fit(X_train, Y_train)

# Predicciones
predicciones = modelo.predict(X_test)

# Precisión
accuracy = accuracy_score(Y_test, predicciones)

print(f'Precisión del modelo: {accuracy}')


