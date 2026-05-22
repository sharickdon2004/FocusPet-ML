import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


class RandomForestModel:

    def __init__(self):

        self.model = None
        self.accuracy = None

        # ================= RUTAS SEGURAS =================
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.csv_path = os.path.join(
            self.base_dir,
            'DataSet_FocusPet.csv'
        )

        self.model_path = os.path.join(
            self.base_dir,
            'randomforest_model.pkl'
        )

        self.cargar_o_entrenar()

    # =====================================================
    # CARGAR O ENTRENAR
    # =====================================================

    def cargar_o_entrenar(self):

        if os.path.exists(self.model_path):

            self.model = joblib.load(self.model_path)

            print(" Random Forest model loaded")

        else:

            print("⚠ Entrenando modelo...")

            self.entrenar()

    # =====================================================
    # ENTRENAMIENTO
    # =====================================================

    def entrenar(self):

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f" No se encontró el dataset:\n{self.csv_path}"
            )

        df = pd.read_csv(self.csv_path)

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

        Y = df['calificacion_beta']

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y,
            test_size=0.2,
            random_state=42
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.model.fit(X_train, Y_train)

        self.accuracy = accuracy_score(
            Y_test,
            self.model.predict(X_test)
        )

        joblib.dump(self.model, self.model_path)

        print(f" Modelo entrenado")
        print(f" Accuracy: {self.accuracy:.2%}")

    # =====================================================
    # PREDICCIÓN
    # =====================================================

    def procesar(self, form):

        try:

            datos = [[
                float(form['edad']),
                float(form['horas_pantalla_pre']),
                float(form['horas_pantalla_post']),
                float(form['sesiones_exitosas']),
                float(form['sesiones_fallidas']),
                float(form['interacciones_rangel']),
                float(form['trivias_ganadas']),
                float(form['recompensas_canjeadas'])
            ]]

            prediccion = self.model.predict(datos)[0]

            return {
                'prediction': int(prediccion),
                'title': self.get_messages(prediccion)['title'],
                'description': self.get_messages(prediccion)['description'],
                'motivation': self.get_messages(prediccion)['motivation'],
                'accuracy': self.accuracy,
                'error': None
            }

        except Exception as e:

            return {
                'error': str(e)
            }

    # =====================================================
    # MENSAJES
    # =====================================================

    def get_messages(self, calificacion):

        mensajes = {

            4: {
                'title': 'Excellent',
                'description': ' Usuario con excelente adaptación digital',
                'motivation': ' Sigue así, eres un ejemplo!'
            },

            3: {
                'title': 'Good',
                'description': ' Buen comportamiento digital',
                'motivation': ' Vas por buen camino!'
            },

            2: {
                'title': 'Regular',
                'description': ' Necesitas mejorar hábitos digitales',
                'motivation': ' Pequeños cambios hacen gran diferencia'
            },

            1: {
                'title': 'Needs Improvement',
                'description': ' Baja adaptación digital',
                'motivation': ' Puedes mejorar mucho más'
            }
        }

        return mensajes.get(calificacion, mensajes[2])


# ================= INSTANCIA =================

model = RandomForestModel()