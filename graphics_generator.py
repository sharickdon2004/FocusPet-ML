import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

os.makedirs(os.path.join('static', 'images'), exist_ok=True)

print("Loading dataset...")
try:
    df = pd.read_csv("DataSet_FocusPet.csv", sep=';', decimal=',')
    if "age" not in df.columns: raise ValueError
except:
    df = pd.read_csv("DataSet_FocusPet.csv")

# ==========================================
# GRÁFICO 1: Rating Distribution Chart
# ==========================================
print("Generating Distribution Chart...")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='beta_rating', palette='viridis')
plt.title('Distribution of Beta Ratings (Target Variable)')
plt.xlabel('Beta Rating (1-5)')
plt.ylabel('Number of Users')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join('static', 'images', 'rating_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()

# ==========================================
# GRÁFICO 2: Correlation Matrix
# ==========================================
print("Generating Correlation Matrix...")
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
# Usamos un mapa de calor (heatmap)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join('static', 'images', 'correlation_matrix.png'), bbox_inches='tight', dpi=150)
plt.close()

# ==========================================
# PREPARACIÓN PARA MATRICES DE CONFUSIÓN
# ==========================================
features = ["age", "screen_time_pre", "screen_time_post", "successful_sessions", 
            "failed_sessions", "rangel_interactions", "trivias_won", "rewards_redeemed"]
X = df[features]
Y = df["beta_rating"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ==========================================
# GRÁFICO 3: DT Confusion Matrix
# ==========================================
print("Generating Confusion Matrix (Decision Tree)...")
dt_model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
dt_model.fit(X_train, Y_train)
Y_pred_dt = dt_model.predict(X_test)

cm_dt = confusion_matrix(Y_test, Y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=dt_model.classes_)

fig, ax = plt.subplots(figsize=(6, 6))
disp_dt.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Decision Tree - Confusion Matrix')
plt.savefig(os.path.join('static', 'images', 'dt_confusion_matrix.png'), bbox_inches='tight', dpi=150)
plt.close()

# ==========================================
# GRÁFICO 4: RF Confusion Matrix
# ==========================================
print("Generating Confusion Matrix (Random Forest)...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)

cm_rf = confusion_matrix(Y_test, Y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)

fig, ax = plt.subplots(figsize=(6, 6))
disp_rf.plot(cmap='Greens', ax=ax, colorbar=False)
plt.title('Random Forest - Confusion Matrix')
plt.savefig(os.path.join('static', 'images', 'rf_confusion_matrix.png'), bbox_inches='tight', dpi=150)
plt.close()

print("All graphics generated and saved in static/images/!")