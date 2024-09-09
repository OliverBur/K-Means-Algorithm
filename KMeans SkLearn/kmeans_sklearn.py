import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier as KNN
color_map = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']

def read_data(file):
    data = pd.read_csv(file)
    return data

def pca(data, n_components=2):
    pca = PCA(n_components)
    data_pca = pca.fit_transform(data)
    data_pca = pd.DataFrame(data_pca, columns=[f'PCA{i+1}' for i in range(n_components)])
    return pca, data_pca

def kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    data['cluster'] = kmeans.labels_
    return data

def plot_clusters(data, n_clusters):
    colors = [color_map[i] for i in data['cluster']]
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=colors)
    plt.title('KMeans Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

def silhouette(data, n_clusters):
    # Se va a calcular el Silhouette Score para determinar el número óptimo de clusters
    silhouette = []
    for i in range(3, 10):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette.append([i, silhouette_score(data, labels)])

    silhouette = pd.DataFrame(silhouette, columns=['i', 'silhouette'])

    plt.plot(silhouette['i'], silhouette['silhouette'], marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.show()

def silhouette_plot(data, n_clusters):
    # Se va a graficar el Silhouette Plot para el número de clusters seleccionado, con la intención de evaluar la calidad de los clusters
    silhouette = silhouette_samples(data.iloc[:, :-1], data['cluster'])  # Ignorar la columna 'cluster' para el cálculo
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = silhouette[data['cluster'] == i]
        cluster_silhouette_values.sort()
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.title('Silhouette Plot for {} Clusters'.format(n_clusters))
    plt.xlabel('Silhouette Score')
    plt.ylabel('Cluster')
    plt.axvline(x=silhouette_score(data.iloc[:, :-1], data['cluster']), color='red', linestyle='--')
    plt.show()

def predict_new_point(pca_model, data_km):
    # Ingresar los datos del nuevo punto, ya sea los predefinidos o manualmente
    nuevo = int(input("\nPredicción de un nuevo punto\n¿Desea ingresar los datos predefinidos o ingresar los datos manualmente? (0 = predefinidos / 1 = manual): "))
    if nuevo == 0:
        new_point = np.array([[13.24, 2.59, 2.87, 21.0, 118.0, 2.8, 2.69, 0.39, 1.82, 4.32, 1.04, 2.93, 735.0]])
    elif nuevo == 1:
        alcohol = float(input("Ingrese el valor de alcohol: "))
        malic_acid = float(input("Ingrese el valor de malic acid: "))
        ash = float(input("Ingrese el valor de ash: "))
        alcalinity_of_ash = float(input("Ingrese el valor de alcalinity of ash: "))
        magnesium = float(input("Ingrese el valor de magnesium: "))
        total_phenols = float(input("Ingrese el valor de total phenols: "))
        flavanoids = float(input("Ingrese el valor de flavanoids: "))
        nonflavanoid_phenols = float(input("Ingrese el valor de nonflavanoid phenols: "))
        proanthocyanins = float(input("Ingrese el valor de proanthocyanins: "))
        color_intensity = float(input("Ingrese el valor de color intensity: "))
        hue = float(input("Ingrese el valor de hue: "))
        od280_od315_of_diluted_wines = float(input("Ingrese el valor de od280/od315 of diluted wines: "))
        proline = float(input("Ingrese el valor de proline: "))
        new_point = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315_of_diluted_wines, proline]])

    # Aplicar el modelo PCA ya entrenado
    new_point_pca = pca_model.transform(new_point)
    
    # Usar KNN para predecir el cluster más cercano
    knn = KNN(n_neighbors=1)
    knn.fit(data_km.iloc[:, :-1], data_km['cluster'])
    cluster = knn.predict(new_point_pca)[0]
    print(f"El punto {new_point} pertenece al cluster {cluster}")
    
    plt.scatter(data_km.iloc[:, 0], data_km.iloc[:, 1], c=data_km['cluster'].apply(lambda x: color_map[x]))
    plt.scatter(new_point_pca[0, 0], new_point_pca[0, 1], s=300, marker='*', edgecolor='black', linewidth=3, label=f'Cluster {cluster}', c=color_map[cluster])
    plt.title('Predicción del nuevo punto')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    

def main():
    print("\nEn este código se implementa un modelo de K-Means para clustering de un conjunto de datos de vinos.\nSe aplica PCA para reducir la dimensionalidad de los datos a 2 dimensiones y se visualizan los clusters resultantes.\nAdemás, se calcula el Silhouette Score para determinar el número óptimo de clusters y se muestra un Silhouette Plot para evaluar la calidad de los clusters.\nFinalmente, se implementa un algoritmo KNN para clasificar un nuevo punto en uno de los clusters obtenidos.")
    data = read_data('wine-clustering.csv')
    pca_model, data_pca = pca(data)
    n_clusters = int(input("\nIngrese el número de clusters (default = 3): "))
    if n_clusters > 10:
        raise ValueError("El número de clusters debe ser menor o igual a 10")
    data_km = kmeans(data_pca, n_clusters)
    plot_clusters(data_pca, n_clusters)
    silhouette(data_pca, n_clusters)
    silhouette_plot(data_km, n_clusters)
    predict_new_point(pca_model, data_km)

if __name__ == '__main__':
    main()