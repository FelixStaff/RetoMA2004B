import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
import networkx as nx
import matplotlib.pyplot as plt
import random
Matrix = np.load("matrixT.npy")
minR = 140
maxR = 199
# Creamos una matriz que solo agarre desde el nodo minR al maxR
newMatrix = np.zeros((maxR-minR, maxR-minR))
for i in range(minR, maxR):
    for j in range(minR, maxR):
        newMatrix[i-minR][j-minR] = Matrix[i][j]
Matrix = newMatrix
Position = np.load("coordenadas.npy") # Aplicamos lo mismo para las coordenadas
newPosition = np.zeros((maxR-minR, 2))
for i in range(minR, maxR):
    newPosition[i-minR] = Position[i]

Position = Position.tolist()
Diccionario = dict(zip(range(len(Position)), Position))
# Para cada valor de nodo, le asignamos un peso intrinseco al nodo
Volumen = []
for i in range(len(Position)):
    # Un valor random entre 1e-6 y 1
    Volumen.append(random.uniform(1e-6, 1))
# Reducir de 199 en 199 a 20 en 20
print (Matrix)
# Los valores de 0 volverlos 10000
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        if Matrix[i][j] == 0:
            Matrix[i][j] = 10000
print (Matrix.shape)
permutation, distance = solve_tsp_local_search(Matrix)
print (permutation,"Tiempo", distance / 3600)
# Imprimir el volumen acumulado
print ("Volumen acumulado", sum(Volumen))
# Crear un grafo
G = nx.Graph()
# Agregar los nodos
G.add_nodes_from(Diccionario.keys())
# Agregar las aristas de la solucion
# Agregamos un nodo de 0 al primero
G.add_edge(0, permutation[0]+minR)
for i in range(len(permutation)-1):
    G.add_edge(permutation[i]+minR, permutation[i+1]+minR)
# Agregar la arista del ultimo al primero
G.add_edge(permutation[-1]+minR, permutation[0]+minR)
# Luego un nodo del ultimo al 0
G.add_edge(permutation[-1]+minR, 0)
# Visualizar el grafo
nx.draw(G, Diccionario, width=1.0, with_labels=True, node_size=80, font_size=5, font_color='black', node_color='pink', edge_color='black', alpha=0.9, arrows=True)
plt.show()