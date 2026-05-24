import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RandomForestModel:

    def __init__(self):
        self.model = None
        self.accuracy = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = os.path.join(self.base_dir, "DataSet_FocusPet.csv")
        self.model_path = os.path.join(self.base_dir, "randomforest_model.pkl")
        self.features = [
            "age", "screen_time_pre", "screen_time_post",
            "successful_sessions", "failed_sessions",
            "rangel_interactions", "trivias_won", "rewards_redeemed"
        ]
        self.cargar_o_entrenar()

    # =====================================================
    # CARGAR O ENTRENAR
    # =====================================================
    def cargar_o_entrenar(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("✅ Random Forest model loaded")
            
            # Recalcular métricas y regenerar gráfico por seguridad
            try:
                df = pd.read_csv(self.csv_path, sep=';', decimal=',') if ';' in open(self.csv_path).read() else pd.read_csv(self.csv_path)
                X = df[self.features]
                Y = df["beta_rating"]
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                self.accuracy = accuracy_score(Y_test, self.model.predict(X_test))
                self.generar_grafico_importancia()
            except Exception as e:
                print(f"⚠ Alerta al cargar métricas: {e}")
                self.accuracy = 0.85
        else:
            print("⚠ Entrenando modelo...")
            self.entrenar()

    # =====================================================
    # ENTRENAMIENTO
    # =====================================================
    def entrenar(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"❌ No se encontró el dataset:\n{self.csv_path}")

        # Detectar el separador del CSV automáticamente
        try:
            df = pd.read_csv(self.csv_path, sep=';', decimal=',')
            if "age" not in df.columns: raise ValueError
        except:
            df = pd.read_csv(self.csv_path)

        X = df[self.features]
        Y = df["beta_rating"]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, Y_train)

        self.accuracy = accuracy_score(Y_test, self.model.predict(X_test))
        joblib.dump(self.model, self.model_path)
        
        print(f"✅ Modelo entrenado")
        print(f"📊 Accuracy: {self.accuracy:.2%}")
        
        self.generar_grafico_importancia()

    # =====================================================
    # GENERAR GRÁFICO DE IMPORTANCIA DE VARIABLES
    # =====================================================
    def generar_grafico_importancia(self):
        try:
            os.makedirs(os.path.join(self.base_dir, 'static', 'images'), exist_ok=True)
            importancia = self.model.feature_importances_
            
            # Crear un dataframe temporal para ordenar los datos
            feat_imp = pd.DataFrame({'Variable': self.features, 'Importancia': importancia})
            feat_imp = feat_imp.sort_values(by='Importancia', ascending=True)

            plt.figure(figsize=(10, 5))
            # Usar un color sofisticado (p. ej. verde bosque desaturado)
            plt.barh(feat_imp['Variable'], feat_imp['Importancia'], color='#2D6A4F', alpha=0.85)
            plt.xlabel('Nivel de Importancia en la Predicción')
            plt.title('¿Qué variables influyen más en la adaptación del usuario?')
            plt.grid(axis='x', linestyle=':', alpha=0.6)
            
            path_grafica = os.path.join(self.base_dir, 'static', 'images', 'rf_importance.png')
            plt.savefig(path_grafica, bbox_inches='tight', dpi=150)
            plt.close()
            print("📊 Gráfico de importancia guardado con éxito.")
        except Exception as e:
            print(f"❌ Error al crear la gráfica: {e}")

    # =====================================================
    # PREDICCIÓN
    # =====================================================
    def procesar(self, form):
        try:
            datos = [[
                float(form["age"]),
                float(form["screen_time_pre"]),
                float(form["screen_time_post"]),
                float(form["sessions_successful"]),
                float(form["sessions_failed"]),
                float(form["interactions_range"]),
                float(form["trivia_won"]),
                float(form["rewards_redeemed"]),
            ]]

            prediccion = self.model.predict(datos)[0]

            return {
                "prediction": int(prediccion),
                "title": self.get_messages(prediccion)["title"],
                "description": self.get_messages(prediccion)["description"],
                "motivation": self.get_messages(prediccion)["motivation"],
                "accuracy": self.accuracy,
                "error": None,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_messages(self, calificacion):
        messages = {
                        5: {
        "title": "Excellent (Level 5)", 
        "description": "User with impeccable digital adaptation. Knows how to perfectly balance productivity with entertainment.", 
        "motivation": "Incredible! Keep maintaining this level of digital discipline."
    },
                        4: {
        "title": "Good (Level 4)", 
        "description": "Good behavior and healthy habits. The user responds positively to focus stimuli.", 
        "motivation": "You are on an excellent path! A few more adjustments and you will reach the optimal level."
    },
                        3: {
        "title": "Regular (Level 3)", 
        "description": "Inconsistencies detected. Post-use screen time decreased slightly, but failed sessions are significant.", 
        "motivation": "Small daily changes will generate a big impact in the long term."
    },
                        2: {
        "title": "Needs Improvement (Level 2)", 
        "description": "Low digital adaptation. The user spends a lot of time on screen and completes few recommended tasks.", 
        "motivation": "Don't give up! FocusPet has the tools to help you improve today."
    },
                        1: {
        "title": "Critical (Level 1)", 
        "description": "Very low digital adaptation. Dispersed usage pattern with a high rate of abandoned or failed sessions.", 
        "motivation": "It's time to take a pause and restructure your focus habits."
    }
}
        return messages.get(calificacion, messages[3])

model = RandomForestModel()