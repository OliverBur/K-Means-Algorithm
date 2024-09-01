import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap

# Se generan 100 datos aleatorios con ruido
data = int(input("Ingrese el número de datos: "))
X, y = datasets.make_moons(data, noise=0.3)

# Se inicializan los centroides aleatoriamente
def centroides(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


# Función para asignar cada punto al centroide más cercano
def asignar_cluster(X, centroids):
    clusters = []
    for point in X:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Actualizar los centroides con la media de los puntos asignados
def actualizar_cluster(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(X[np.random.choice(X.shape[0])])
    return np.array(new_centroids)

# Función principal para realizar las iteraciones
def kmeans(X, k, max_iters=100):
    centroids = centroides(X, k)
    for i in range(max_iters):
        clusters = asignar_cluster(X, centroids)
        new_centroids = actualizar_cluster(X, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Implementación manual de KNN para clasificación de un nuevo punto
def knn(X, y, query_point, k=3):
    # Calcula la distancia entre el punto de consulta y todos los puntos en X
    distances = []
    for i in range(len(X)):
        dist = euclidean_distance(X[i], query_point)
        distances.append((dist, y[i]))
    
    # Ordena las distancias y selecciona los k vecinos más cercanos
    distances = sorted(distances)[:k]
    
    # Toma las etiquetas de los k vecinos más cercanos
    neighbors = [label for _, label in distances]
    
    # Cuenta cuántos vecinos pertenecen a cada clase
    class_counts = {0: neighbors.count(0), 1: neighbors.count(1)}
    
    # Predice la clase del punto de consulta
    prediction = max(set(neighbors), key=neighbors.count)
    
    return prediction, class_counts

# Parámetro para el número de clusters
k = int(input("Ingrese el número de clusters: ")) 
if k > 10:
    raise ValueError("El número de clusters debe ser menor o igual a 10")

# Ejecutar K-Means
centroids, clusters = kmeans(X, k)

# Crear la paleta de colores para visualizar los clusters y que se indique el numero de puntos en cada cluster por color
color_map = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
color_names = ['Azul', 'Rojo', 'Verde', 'Amarillo', 'Morado', 'Naranja', 'Marrón', 'Rosado', 'Gris', 'Cian']
custom_cmap = ListedColormap(color_map[:k])

# Visualización de los clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap=custom_cmap)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='x')
plt.title('K-Means Clustering')
print("\nNúmero de puntos en cada cluster:")

# Imprimir cuántos puntos hay en cada cluster
for i in range(k):
    color_name = color_names[i % len(color_names)]  # Asigna un nombre de color según el índice
    cluster_count = np.sum(clusters == i)
    print(f"Cluster {color_name}: {cluster_count} puntos")


# Agregar un input con un nuevo valor y usar knn para predecir a qué cluster pertenece
print("\nIngrese las coordenadas de un punto para predecir a qué cluster pertenece:")
punto_a_predecir = np.array([float(input("Valor de x [-1 , 1]: ")), float(input("Valor de y [-1 , 1]: "))])
cluster_punto, _ = knn(centroids, np.arange(k), punto_a_predecir, k=1)
color_name = color_names[cluster_punto % len(color_names)]
print(f"\nEl punto {punto_a_predecir} pertenece al Cluster {color_name}")

plt.scatter(punto_a_predecir[0], punto_a_predecir[1], s=300, marker='*', 
            edgecolor='black', linewidth=3,
            c=color_map[cluster_punto], label=f'Cluster {color_name}')

plt.show()