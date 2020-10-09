'''
/////////////////////////////////////////////
//    PONTIFICIA UNIVERSIDAD JAVERIANA     //
//                                         //
//  Carlos Cadena y Carlos Redondo         //
//  Procesamiento de imagenes y vision     //
//  TALLER #2                              //
/////////////////////////////////////////////
'''

import cv2
import numpy as np
import math

# Respuesta del mouse
def OnMouseAction1(event, x, y, flags, param):
       # img1 = img.copy() #Image deep copy
       global puntos1
       if event == cv2.EVENT_LBUTTONDBLCLK:  # Doble clic izquierdo para el evento
              cv2.circle(imagen, (x, y), 3, (255, 0, 0), -1)  # Dibujar circulo
              puntos1.append([x, y]) # vector de coordenadas
              #print(puntos1)

puntos1 = []
pts1 = np.float32([])
pts2 = np.float32([])

path1 = input("Por favor ingrese la ruta de la imagen: ") # Dirección de la imagen 1
#img1 = cv2.imread('C:/Users/Daniel/Documents/U/lena.png')
img1 = cv2.imread(path1)    # lectura de la imagen 1
imagen = img1               # Imagen 1
img1_c = img1.copy()        # Copia de la Imagen 1
img1_c2 = img1.copy()
cv2.namedWindow('image 1')
cv2.setMouseCallback('image 1', OnMouseAction1)  # Función para ingresar puntos

# Primera imagen
while (1):
       if len(puntos1) == 3:
              pts1 = np.float32([puntos1]) # Almacenar las coordenadas
       cv2.imshow('image 1', img1)        # Mostrar imagen
       k = cv2.waitKey(1) & len(pts1)
       if k == 1:                         # Condición para cerrar la imagen
              puntos1 = []                # Limpieza del vector de coordenadas
              cv2.destroyAllWindows()     # Cerrar imagen
              break                       # Salir del ciclo

# segunda imagen
path2 = input("Por favor ingrese la ruta de la otra imagen: ") # Dirección de la imagen 2
#img2 = cv2.imread('C:/Users/Daniel/Documents/U/lena_warped.png')
img2 = cv2.imread(path2)    # lectura de la imagen 2
imagen = img2               # Imagen 2
cv2.namedWindow('image 2')
cv2.setMouseCallback('image 2', OnMouseAction1) # Función para ingresar puntos

while (1):
       if len(puntos1) == 3:
              pts2 = np.float32([puntos1]) # Almacenar las coordenadas
       cv2.imshow('image 2', img2)         # Mostrar imagen
       k = cv2.waitKey(1) & len(pts2)
       if k == 1:                         # Condición para cerrar la imagen
              puntos1 = []                # Limpieza del vector de coordenadas
              cv2.destroyAllWindows()     # Cerrar imagen
              break                       # Salir del ciclo


# Transformada Affine
M_affine = cv2.getAffineTransform(pts1, pts2)    # Matriz afin
image_affine = cv2.warpAffine(img1_c, M_affine, img1_c.shape[:2]) # Transformada afin
print(f'Esta es la matriz de la transformada afin:\n {M_affine}') # Impresion de la matriz de la Transformada afin
cv2.imshow("Image", image_affine) # Visualización de la imagen con la transformada afin

# computo de los parametros

# Parametros de escala
s11 = (M_affine[0, 0])      # Coordenadas 00 de la matriz
s21 = (M_affine[1, 0])      # Coordenadas 10 de la matriz
s0 = np.sqrt((s11*2)+(s21*2)) # parametro de escala s0

s12 = (M_affine[0, 1])      # Coordenadas 01 de la matriz
s22 = (M_affine[1, 1])      # Coordenadas 11 de la matriz
s1 = np.sqrt((s12*2)+(s22*2)) # parametro de escala s1

# Parametros de rotacion
theta = math.atan(s21/s11)  # angulo de rotación en radianes
theta_g = theta * 180/ np.pi # angulo de rotación en grados

# Parametros de traslacion
s13 = (M_affine[0, 2])      # Coordenadas 02 de la matriz
s23 = (M_affine[1, 2])      # Coordenadas 12 de la matriz
x0 = (s13*np.cos(theta_g)-s23*np.sin(theta_g))/s0 # parametro de traslación x0
x1 = (s13*np.cos(theta_g)-s23*np.sin(theta_g))/s1 # parametro de traslación x1

# Aproximacion por Matriz de similitud
M_simil = np.float32([[s0*np.cos(theta), s1*np.sin(theta), (s0*x0*np.cos(theta))+(s1*x1*np.sin(theta))],
                     [-s0*np.sin(theta), s1*np.cos(theta), (s1*x1*np.cos(theta))-(s0*x0*np.sin(theta))],
                     ])

image_similarity = cv2.warpAffine(img1_c2, M_simil, img1_c2.shape[:2]) # Transformada de similitud sobre img1
cv2.imshow("Imagen Similitud", image_similarity) # Mostrar imagen son aproximación de similitud

# Calculo del error
N_m = np.append(pts1[0].transpose(), np.array([[1, 1, 1]]), axis=0) # Matriz de puntos de img 1
puntos_trans = M_simil.dot(N_m)  # multiplicación de las matrices de puntos y similitud = A
puntos_trans = puntos_trans.transpose() # A transpuesta

error0 = np.linalg.norm(puntos_trans - pts2[0], axis=1)  # Cálculo del error
#error = abs(puntos_trans-pts2) # Cálculo del error
#print(error)
print(f'Este es el error con respecto a los puntos tomados de la img2:\n {error0}') # Impresión del error

cv2.waitKey(0)
