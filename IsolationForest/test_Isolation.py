# Ejemplo de uso del algoritmo Isolation Forest implementado
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from IsolationForest import IsolationForest

# Crear datos sintéticos (normales)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=42)

# Agregar algunas anomalías artificiales
anomalies = np.random.uniform(low=-6, high=6, size=(20, 2))
X_with_anomalies = np.vstack([X, anomalies])

# Instanciar y entrenar el modelo
model = IsolationForest(n_estimators=100, subsample_size=256, max_height=8, random_state=42, n_jobs=1, contamination=0.07)
model.fit(X_with_anomalies)

# Obtener puntuaciones y predicciones
scores = model.decision_function(X_with_anomalies)
labels = model.predict(X_with_anomalies)

# Visualización
plt.figure(figsize=(8, 6))
plt.scatter(X_with_anomalies[labels == 0][:, 0], X_with_anomalies[labels == 0][:, 1], 
            c='blue', label='Normal', s=30)
plt.scatter(X_with_anomalies[labels == 1][:, 0], X_with_anomalies[labels == 1][:, 1], 
            c='red', label='Anomalía', s=40, marker='x')
plt.title("Detección de Anomalías con Isolation Forest con contaminación {}%".format(model.contamination * 100))
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IsolationForest_test.png")
# Guardar el gráfico
plt.show()