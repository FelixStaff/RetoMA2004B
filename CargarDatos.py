import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Cargar_Datos():
    # Creamos el diccionario con los datos
    Datos = {
        'DistribucionC': {"valor": [], "probabilidad": []},
        'DistribucionP': {"valor": [], "probabilidad": []},
        'VolumenesP': {"Producto": {}}
    }
    # Cargamos los datos y volver la primera fila en int
    DistribucionC = pd.read_csv('Datos/DistC.csv').values
    Datos['DistribucionC']['valor'] = DistribucionC[:, 0].astype(int)
    Datos['DistribucionC']['probabilidad'] = DistribucionC[:, 1]
    # Cargamos los datos y volver la primera fila en int
    DistribucionP = pd.read_csv('Datos/DistP.csv').values
    Datos['DistribucionP']['valor'] = DistribucionP[:, 0].astype(int)
    Datos['DistribucionP']['probabilidad'] = DistribucionP[:, 1]
    # Cargamos los datos y volver la primera fila en int
    VolumenesP = pd.read_csv('Datos/infoP.csv').values
    for i in range(len(VolumenesP)):
        Datos['VolumenesP']['Producto'][VolumenesP[i, 0]] = VolumenesP[i, 1]
    return Datos

# Generamos la funcion que genera un pedido
def Generar_Pedido(Datos):
    # Creamos el diccionario con los datos
    Pedido = {
        'Volumen': [],
        'Cantidad': [],
    }
    # Generamos la cantidad de productos
    Cantidad = np.random.choice(Datos['DistribucionC']['valor'], p=Datos['DistribucionC']['probabilidad'])
    # Generamos los productos
    for i in range(Cantidad):
        Producto = np.random.choice(Datos['DistribucionP']['valor'], p=Datos['DistribucionP']['probabilidad'])
        Pedido['Volumen'].append(Datos['VolumenesP']['Producto'][Producto])
        Pedido['Cantidad'].append(Producto)
    # Ahora solo sumamos los volumenes
    Volumen = 0
    for i in Pedido['Volumen']:
        Volumen += i
    return Volumen

# Creamos el if
if __name__ == '__main__':

    Datos = Cargar_Datos()
    for i in range(10):
        try:
            Pedido = Generar_Pedido(Datos)
            print ("Pedido:",Pedido)
        except:
            print("Error en la generacion del pedido:", i)
    