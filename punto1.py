'''
/////////////////////////////////////////////
//    PONTIFICIA UNIVERSIDAD JAVERIANA     //
//                                         //
//  Carlos Cadena y Carlos Redondo         //
//  Procesamiento de imagenes y vision     //
//  TALLER #2                              //
/////////////////////////////////////////////
'''

'''En colaboración con parte del código suministrado por el Ing. Julian Armando Quiroga Sepulveda'''

import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time


def segcolors(path, metodo):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    suma_parcial = 0
    sumas = []
    n_colors = 0
    method = ['kmeans', 'gmm']
    select = 0

    image = np.array(image, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    rows, cols, ch = image.shape
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch))

    t0 = time()
    image_array_sample = shuffle(image_array, random_state=0)[:10000]

    for i in range(10):
        n_colors = n_colors + 1
        if metodo == 'gmm':
            model = GMM(n_components=n_colors).fit(image_array_sample)
        else:
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        t0 = time()
        if metodo == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_
        else:
            labels = model.predict(image_array)
            centers = model.cluster_centers_

        suma_parcial = 0
        suma = 0

        for index in range(0, image_array.shape[0]):
            suma_parcial = abs(image_array[index] - centers[labels[index]])
            suma += np.linalg.norm(suma_parcial)

        sumas.append(suma)

    plt.figure(1)
    plt.title(f'Distancia Intra-Cluster con {metodo} ')
    plt.ylabel('Distancia Euclideana')
    plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], (sumas))
    plt.show()
    print(sumas)


path1 = input("Por favor ingrese la ruta de la imagen: ")
metodo = input("Por favor ingrese el método: ")
segmentacion = segcolors(path1, metodo)
