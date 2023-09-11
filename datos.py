import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargamos los datos
def main():

    # Cargamos los datos
    data = pd.read_csv('199Distancia.csv', sep=';')
    # Eliminamos los datos que no nos sirven
    print(data.head())
    # Creamos la matriz de distancias
    matrix = np.zeros((max(data['from'])+1, max(data['to'])+1))
    matrixT = np.zeros((max(data['from'])+1, max(data['to'])+1))
    # Llenar con -1
    matrix.fill(-1)
    matrixT.fill(-1)
    # Llenamos la matriz de distancias
    for i in range(len(data)):
        fr, to = data['from'][i], data['to'][i]
        matrix[fr][to] = data['distancia'][i]
        tiempo = data['tiempo'][i].split(':')
        tiempo = int(tiempo[0])*3600 + int(tiempo[1])*60
        matrixT[fr][to] = tiempo
    # Contamos la cantdad de -1
    count = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == -1:
                count += 1
    # Media
    print("Media", np.mean(matrix))
    print("Desviacion estandar", np.std(matrix))
    print("Cantidad de -1", count)
    print("Tamanio de la matriz", matrix.shape)
    # Graficamos la matriz
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()
    # Sumamos todos los valores de los tiempos y lo dividimos por la cantidad de valores
    suma = 0
    for i in range(len(data)):
        # El tiempo esta en 00:00:00
        # Pasarlo a segundos
        tiempo = data['tiempo'][i].split(':')
        tiempo = int(tiempo[0])*3600 + int(tiempo[1])*60 
        suma += tiempo
    suma = suma/len(matrix) * 1.3
    print("Promedio de tiempos", suma)
    # Dividimos el resultado por 8 horas en segundos
    suma = suma/(8*3600)
    print("Promedio de tiempos en 8 horas", suma)
    # Guardamos en un npy
    print (matrix.shape)
    np.save('matrix.npy', matrix)
    np.save('matrixT.npy', matrixT)
if __name__ == "__main__":
    main()