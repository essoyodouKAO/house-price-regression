import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- FONCTIONS ---

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradient_descent(X, y, w, b, learning_rate):
    n = len(y)
    y_pred = np.dot(X, w) + b
    dw = (-2/n) * np.dot(X.T, (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# --- PRÉPARATION DES DONNÉES ---

df = pd.read_csv(r"C:\Users\serge\Downloads\archive\Housing.csv")

# Encodage des données textuelles
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_columns:
    df[col] = df[col].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

y = df['price'].values
X = df.drop('price', axis=1).values

# Split Train/Test (80/20)
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.8 * len(indices))

X_train_raw, X_test_raw = X[indices[:split]], X[indices[split:]]
y_train_raw, y_test_raw = y[indices[:split]], y[indices[split:]]

# Normalisation (Z-score)
mean_X, std_X = np.mean(X_train_raw, axis=0), np.std(X_train_raw, axis=0)
X_train, X_test = (X_train_raw - mean_X) / std_X, (X_test_raw - mean_X) / std_X

mean_y, std_y = np.mean(y_train_raw), np.std(y_train_raw)
y_train, y_test = (y_train_raw - mean_y) / std_y, (y_test_raw - mean_y) / std_y

# --- ENTRAÎNEMENT ---

w = np.zeros(X_train.shape[1]) 
b = 0
learning_rate = 0.01
epochs = 1000
mse_history = [] # Pour stocker l'évolution de l'erreur

for i in range(epochs):
    w, b = gradient_descent(X_train, y_train, w, b, learning_rate)
    loss = compute_mse(y_train, np.dot(X_train, w) + b)
    mse_history.append(loss)
    
    if i % 200 == 0:
        print(f"Époque {i}: MSE = {loss:.4f}")

# --- RÉSULTATS & COMPARAISON ---

# Modèle Custom
y_pred_custom = np.dot(X_test, w) + b
res = calculate_metrics(y_test, y_pred_custom)

# Modèle Scikit-Learn
model_sk = LinearRegression()
model_sk.fit(X_train, y_train)
y_pred_sk = model_sk.predict(X_test)

print("\n--- COMPARAISON ---")
print(f"Custom  | R2: {res['R2']:.4f} | MSE: {res['RMSE']**2:.4f}")
print(f"Sklearn | R2: {r2_score(y_test, y_pred_sk):.4f} | MSE: {mean_squared_error(y_test, y_pred_sk):.4f}")

# --- VISUALISATION ---

plt.figure(figsize=(12, 5))

# Graphique 1 : Courbe d'apprentissage
plt.subplot(1, 2, 1)
plt.plot(range(epochs), mse_history, color='blue', lw=2)
plt.title('Évolution de la fonction de coût (MSE)')
plt.xlabel('Époques')
plt.ylabel('Erreur (MSE)')
plt.grid(True, linestyle='--')

# Graphique 2 : Prédictions vs Réalité
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_custom, alpha=0.5, color='green', label='Prédictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Parfait')
plt.title('Prédictions vs Réalité (Test Set)')
plt.xlabel('Valeurs Réelles (Normalisées)')
plt.ylabel('Valeurs Prédites (Normalisées)')
plt.legend()
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.show