import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

class FocusPetModel:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.cargar_o_entrenar_modelo()
    
    def cargar_o_entrenar_modelo(self):
        """Loads the model if exists, otherwise trains it"""
        if os.path.exists('focuspet_model.pkl'):
            self.model = joblib.load('focuspet_model.pkl')
            print(" Model loaded from disk")
        else:
            self.entrenar_modelo()
    
    def entrenar_modelo(self):
        """Trains the decision tree model"""
        df = pd.read_csv('DataSet_FocusPet.csv')
        
        X = df[[
            'age',
            'screen_time_pre',
            'screen_time_post',
            'successful_sessions',
            'failed_sessions',
            'rangel_interactions',
            'trivias_won',
            'rewards_redeemed'
        ]]
        
        Y = df['beta_rating']
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        self.model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, Y_train)
        
        predicciones = self.model.predict(X_test)
        self.accuracy = accuracy_score(Y_test, predicciones)
        
        joblib.dump(self.model, 'focuspet_model.pkl')
        
        print(f' Model trained and saved')
        print(f' Model accuracy: {self.accuracy:.2%}')
    
    def predict(self, datos_usuario):
        """Makes a prediction"""
        if isinstance(datos_usuario, list) and len(datos_usuario) > 0:
            if not isinstance(datos_usuario[0], list):
                datos_usuario = [datos_usuario]
        
        prediccion = self.model.predict(datos_usuario)[0]
        return int(prediccion)
    
    def get_messages(self, calificacion):
        """Returns personalized messages based on rating"""
        messages = {
            4: {
                'title': 'Excellent',
                'description': ' Congratulations! You are a FocusPet star user. Your commitment to digital time management is exemplary and you are making the most of all features.',
                'motivation': ' Keep it up! You are a role model for others. Share your experience and help others improve their digital habits.'
            },
            3: {
                'title': 'Good',
                'description': ' You are on the right track. FocusPet is having a positive impact on your digital habits, although there are still areas for improvement.',
                'motivation': ' Don\'t stop! Every day is an opportunity to improve. Try to increase your interactions with Rangel.'
            },
            2: {
                'title': 'Regular',
                'description': ' You have started using FocusPet, but the results are not consistent yet. It\'s time to commit more to the platform.',
                'motivation': ' Small changes lead to big results. Start with a daily goal of reducing screen time by 30 minutes.'
            },
            1: {
                'title': 'Needs Improvement',
                'description': ' FocusPet can help you more if you dedicate time and consistency. You are not taking full advantage of its potential.',
                'motivation': ' It\'s never too late to start! Set daily reminders and participate in more trivia to improve your experience.'
            }
        }
        return messages.get(calificacion, messages[2])
    
    def get_accuracy(self):
        """Returns model accuracy"""
        return self.accuracy

# Create a single instance of the model
model = FocusPetModel()