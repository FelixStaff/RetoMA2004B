import numpy as np
from python_tsp.heuristics import solve_tsp_local_search
import networkx as nx
import matplotlib.pyplot as plt
import random
import vrpy

Matrix = np.load("matrixT.npy")
minR = 0
maxR = 60
# Creamos una matriz que solo agarre desde el nodo minR al maxR
newMatrix = np.zeros((maxR-minR, maxR-minR))
for i in range(minR, maxR):
    for j in range(minR, maxR):
        newMatrix[i-minR][j-minR] = Matrix[i][j]
Matrix = newMatrix
Position = np.load("coordenadas.npy") # Aplicamos lo mismo para las coordenadas
Position = Position.tolist()
newPosition = np.zeros((maxR-minR, 2))
for i in range(minR, maxR):
    newPosition[i-minR] = Position[i]

Position = newPosition
Diccionario = dict(zip(range(len(Position)), Position))
# Cambiamos el nodo 0 por Source y el nodo 1 por Sink
Diccionario["Source"] = Diccionario.pop(0)
Diccionario["Sink"] = Diccionario.pop(1)
# Para cada valor de nodo, le asignamos un peso intrinseco al nodo
Volumen = []
# Creamos un grafo
G = nx.DiGraph()
# Agregamos los nodos
Nodos = []
for i in range(len(Position)):
    if i == 0:
        Nodos.append("Source")
    elif i == 1:
        Nodos.append("Sink")
    else:
        Nodos.append(i+minR)
G.add_nodes_from(Nodos)
# Agregamos las aristas
for i in range(len(Matrix)):
    for j in range(len(Matrix)):
        # Si es el nodo 0 es Source, si es el final es Sink
        if i == 0:
            G.add_edge("Source", j+minR, cost=Matrix[i][j])
        elif j == 0:
            G.add_edge(i+minR, "Sink", cost=Matrix[i][j])
        else:
            G.add_edge(i+minR, j+minR, cost=Matrix[i][j])
# Se crea el problema
prob = vrpy.VehicleRoutingProblem(G, 1, 8*3600)
print (prob.best_routes)