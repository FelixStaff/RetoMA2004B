import pandas
import numpy as np
import matplotlib.pyplot as plt

# Cargamos los datos
def main():
    # Cargamos los datos
    data = pandas.read_csv('199Coordenadas.csv', sep=';', encoding='latin-1')
    # Eliminamos los datos que no nos sirven
    print(data.head())
    # Visualizamos en un grafico la Latitud y Longitud de cada Nodo
    # Agregarles el numero de nodo
    plt.scatter(data['Longitud'],data['Latitud'], s=10)
    plt.xlabel('Latitud')
    plt.ylabel('Longitud')
    plt.title('Grafico de Latitud y Longitud')
    for i in range(len(data)):
        plt.annotate(data["Nodo"][i], (data['Longitud'][i],data['Latitud'][i]))

    plt.show()
    # Borramos la columna de Nodo y Direccion
    data = data.drop(['Nodo', 'Direccion'], axis=1)
    # Convertimos a un array
    data = data.values
    # Intercambiar las columnas para que sea Latitud, Longitud
    data = np.array([data[:,1], data[:,0]])
    data = data.T
    # Lo guardamos en un npy
    print(data)
    print(data.shape)
    np.save('coordenadas.npy', data)

if __name__ == "__main__":
    main()