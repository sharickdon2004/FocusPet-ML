import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class FocusPetModel:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.classification_report = None
        self.cargar_o_entrenar_modelo()
    
    def cargar_o_entrenar_modelo(self):
        """Loads the model if exists, otherwise trains it"""
        if os.path.exists('focuspet_rf_model.pkl'):
            # Cargar modelo existente
            self.model = joblib.load('focuspet_rf_model.pkl')
            print(" Random Forest model loaded from disk")
        else:
            # Entrenar nuevo modelo
            self.entrenar_modelo()
    
    def entrenar_modelo(self):
        """Trains the Random Forest model"""
        # Cargar dataset
        df = pd.read_csv('Dataset_Beta_FocusPet_V2 - Dataset_Beta_FocusPet_V2.csv')
        
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
            X, Y, test_size=0.2, random_state=42
        )
        
        # Crear modelo Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,           # Número de árboles
            criterion='gini',            # Criterio de división
            max_depth=10,                # Profundidad máxima
            min_samples_split=5,         # Muestras mínimas para dividir
            min_samples_leaf=2,          # Muestras mínimas en hoja
            max_features='sqrt',         # Características por árbol
            bootstrap=True,              # Muestreo con reemplazo
            oob_score=True,              # Out-of-bag score
            random_state=42,
            n_jobs=-1                    # Usar todos los núcleos
        )
        
        # Entrenar
        self.model.fit(X_train, Y_train)
        
        # Predicciones y precisión
        predicciones = self.model.predict(X_test)
        self.accuracy = accuracy_score(Y_test, predicciones)
        self.classification_report = classification_report(Y_test, predicciones)
        
        # Guardar modelo
        joblib.dump(self.model, 'focuspet_rf_model.pkl')
        
        print(f'Random Forest model trained and saved')
        print(f'Model accuracy: {self.accuracy:.2%}')
        print(f' Number of trees: {self.model.n_estimators}')
        print(f' OOB Score: {self.model.oob_score_:.2%}')
    
    def predict(self, datos_usuario):
        """Makes a prediction"""
        if isinstance(datos_usuario, list) and len(datos_usuario) > 0:
            if not isinstance(datos_usuario[0], list):
                datos_usuario = [datos_usuario]
        
        prediccion = self.model.predict(datos_usuario)[0]
        
        # Obtener probabilidades de cada clase
        probabilidades = self.model.predict_proba(datos_usuario)[0]
        
        return int(prediccion), probabilidades
    
    def predict_single(self, datos_usuario):
        """Returns only the prediction (compatibility)"""
        prediccion, _ = self.predict(datos_usuario)
        return prediccion
    
    def get_feature_importance(self):
        """Returns feature importance for the model"""
        features = [
            'Age',
            'Screen hours (PRE)',
            'Screen hours (POST)',
            'Successful sessions',
            'Failed sessions',
            'Interactions with Rangel',
            'Trivias won',
            'Rewards redeemed'
        ]
        importancia = self.model.feature_importances_
        
        # Ordenar por importancia
        feature_importance_list = sorted(
            zip(features, importancia),
            key=lambda x: x[1],
            reverse=True
        )
        return feature_importance_list
    
    def get_messages(self, calificacion):
        """Returns personalized messages based on rating"""
        messages = {
            4: {
                'title': 'Excellent',
                'description': ' Congratulations! You are a FocusPet star user. Your commitment to digital time management is exemplary and you are making the most of all features. The Random Forest model predicts you will continue with excellent habits.',
                'motivation': ' Keep it up! You are a role model for others. Share your experience and help others improve their digital habits.'
            },
            3: {
                'title': 'Good',
                'description': ' You are on the right track. FocusPet is having a positive impact on your digital habits, although there are still areas for improvement. The model sees great potential in your behavior pattern.',
                'motivation': ' Don\'t stop! Every day is an opportunity to improve. Try to increase your interactions with Rangel.'
            },
            2: {
                'title': 'Regular',
                'description': ' You have started using FocusPet, but the results are not consistent yet. The Random Forest model suggests that with more commitment, you could significantly improve your rating.',
                'motivation': ' Small changes lead to big results. Start with a daily goal of reducing screen time by 30 minutes.'
            },
            1: {
                'title': 'Needs Improvement',
                'description': ' FocusPet can help you more if you dedicate time and consistency. The model indicates that you are not yet taking full advantage of the platform\'s potential.',
                'motivation': ' It\'s never too late to start! Set daily reminders and participate in more trivia to improve your experience.'
            }
        }
        return messages.get(calificacion, messages[2])
    
    def get_accuracy(self):
        """Returns model accuracy"""
        return self.accuracy
    
    def get_model_info(self):
        """Returns model information"""
        return {
            'type': 'Random Forest Classifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'criterion': self.model.criterion,
            'accuracy': self.accuracy,
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        }

# Create a single instance of the model
model = FocusPetModel()