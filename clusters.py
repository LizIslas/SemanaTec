import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Crear un DataFrame de ejemplo
data = {
    'Edad': [23, 25, 26, 24, 30, 24, 22, 28, 27, 24],
    'Salario': [48000, 52000, 58000, 49000, 60000, 48000, 50000, 62000, 59000, 54000]
}
df = pd.DataFrame(data)

# Aplicar el algoritmo K-Means
# Inicializamos el algoritmo con el número de clusters que deseamos. En este caso, 3 clusters.
kmeans = KMeans(n_clusters=3, random_state=0)

# Ajustar el algoritmo a nuestros datos y obtener las etiquetas de los clusters
df['Cluster'] = kmeans.fit_predict(df[['Edad', 'Salario']])

# Mostrar los resultados de clustering
print(df)

# Visualización de los clusters
plt.scatter(df['Edad'], df['Salario'], c=df['Cluster'], cmap='viridis')

# Mostrar los centros de los clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')

plt.xlabel('Edad')
plt.ylabel('Salario')
plt.title('Clustering K-Means')
plt.show()