import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import math
import time
import os

from CargarDatos import Generar_Pedido, Cargar_Datos

# Clase de grafos de viaje
class GrafoViaje:

    def __init__(self, Matrix : np.ndarray = None, Position : np.ndarray = None, limite : int = 5000, bordes : int = 2):
        plt.rcParams["figure.figsize"] = [7.50, 5.50]
        plt.rcParams["figure.autolayout"] = True
        self.Matrix = Matrix
        self.Position = Position.tolist()
        self.Diccionario = {}
        self.limite = limite
        self.bordes = bordes
        self.Graph = nx.Graph()
        self.Nodes = {}
        self.Egdes = []
        self.Nombres = []
        for i in range(len(Matrix)):
            self.Nodes[i] = "Ciudad " + str(i)
            self.Nombres.append("Ciudad " + str(i))
        for i in range(len(Matrix)):
            for j in range(len(Matrix)):
                if Matrix[i][j] != 0:
                    if (i != j and Matrix[i][j] <= limite) or j == 0 or i == 0:
                            self.Egdes.append((i, j, Matrix[i][j]))
        self.Graph.add_nodes_from(self.Nodes)
        self.Graph.add_weighted_edges_from(self.Egdes)
        # Calculamos la posicion como un diccionario
        self.Diccionario = dict(zip(self.Nodes, self.Position))
        pass

    def ReduceGraph(self):
        # Para esta funcion, vemos los grafos que esten muy cerca unos de otros como uno solo
        # Para esto, vamos a ver los nodos que esten a una distancia menor a 7
        # Si la distancia es menor a 7, los vamos a unir en un solo nodo
        Nodes = self.Graph.nodes()
        Positiones = self.Position
        # Lo vovemos un diccionario
        Positiones = dict(zip(Nodes, Positiones))
        NewPositione = {}
        print("Numero de nodos",len(Nodes))
        Erased = []
        for i in list(Nodes):
            # Incluimos los nodos que esten dentro de i
            InsideNodes = []
            ExtraWeight = 0
            Name = str(i)
            for j in list(Nodes):
                if i != j:
                    if self.Graph.has_edge(i, j):
                        if self.Graph[i][j]['weight'] <= self.bordes and self.Graph[i][j]['weight'] >= 0:
                            if j not in Erased:
                                InsideNodes.append(j)
                                Erased.append(j)
                                Name += "," +str(j) 
                                # Agregamos el peso extra
                                ExtraWeight += self.Graph[i][j]['weight']
            if len(InsideNodes) > 0:
                # Creamos el nuevo nodo
                NewNode = Name
                # Agregamos el nodo al grafo
                self.Graph.add_node(NewNode)
                # Agregamos la posicion
                NewPositione[NewNode] = Positiones[i]
                # Agregamos los nodos adentro, pero los nodos de los eliminados
                Edges = []
                for j in InsideNodes:
                    for k in list(Nodes):
                        if j != k:
                            if self.Graph.has_edge(j, k):
                                # checar si k esta en InsideNodes, en caso de que si, vemos cual es el menor
                                if k not in InsideNodes:
                                    # A"nadir el peso extra de la arista de ExtraWeight
                                    weight = self.Graph[j][k]['weight'] + ExtraWeight
                                    Edges.append((NewNode, k, weight))
                                    Edges.append((k, NewNode, weight))
                self.Graph.add_weighted_edges_from(Edges)
                # Eliminamos los nodos viejos
                for j in InsideNodes:
                    self.Graph.remove_node(j)
                # Eliminamos el nodo viejo
                self.Graph.remove_node(i)
                # Agregamos el nodo nuevo
        # Agregamos los nodos que faltan al diccionario
        for i in list(self.Graph.nodes()):
            if i not in NewPositione:
                NewPositione[i] = self.Diccionario[i]
        # Actualizamos el diccionario
        self.Diccionario = NewPositione
        # Actualizamos la posicion
        self.Position = []
        for i in list(self.Graph.nodes()):
            self.Position.append(self.Diccionario[i])
        print("Numero de nodos",len(self.Graph.nodes()))
        pass

    def GetMatrix(self):
        return self.Matrix
    
    def CalculateMatrix(self):
        # Calculamos la matriz de adyacencia
        # Para esto, vamos a calcular la distancia entre los nodos
        NewMatrix = np.zeros((len(self.Graph.nodes()), len(self.Graph.nodes())))
        listaNodos = list(self.Graph.nodes())
        for i in range(len(self.Graph.nodes())):
            for j in range(len(self.Graph.nodes())):
                if i != j:
                    # Imprimimos el nombre de los nodos
                    if self.Graph.has_edge(listaNodos[i], listaNodos[j]):
                        NewMatrix[i][j] = self.Graph[listaNodos[i]][listaNodos[j]]['weight']
                    else:
                        NewMatrix[i][j] = -1
                else:
                    NewMatrix[i][j] = -1
        return NewMatrix

    def ShowGraph(self):
        nx.draw(self.Graph, with_labels=True, width=.5, node_size=100, font_size=7, font_color='black', node_color='pink', edge_color='black', alpha=0.9)
        # Guardamos la imagen
        plt.savefig("Grafo.png")
        plt.show()
        pass

    def ShowRealGraph(self, name="GrafoReal.png"):
        # Este grafo si toma en cuenta las distancias reales usando las posiciones
        # Creamos el grafo
        # Dibujamos el grafo pero ahora usando las posiciones
        print (len(self.Position))
        position = dict(zip(self.Graph.nodes(), self.Position))
        print (len(self.Graph.nodes()))
        nx.draw(self.Graph, position, with_labels=True, width=.1, node_size=80, font_size=5, font_color='black', node_color='pink', edge_color='black', alpha=0.9)
        # Guardamos la imagen
        plt.savefig("GrafoReal.png", dpi=1000)
        plt.show()

    def ShowWithPosition(self):
        # Dibujamos el grafo pero ahora usando las posiciones
        nx.draw(self.Graph, with_labels=True, width=.1, node_size=80, font_size=5, font_color='black', node_color='pink', edge_color='black', alpha=0.9)
        plt.show()

    def GoTo(self, start, end):
        # Vamos a buscar el camino mas corto entre dos nodos
        
        path = nx.dijkstra_path(self.Graph, start, end)
        print(path)
        # Vamos a calcular la distancia
        distance = 0
        for i in range(len(path)-1):
            distance += self.Graph[path[i]][path[i+1]]['weight']
        print("Distancia",distance)

def GenerarGrafo(pathMatrix="matrix.npy",pathPosition="coordenadas.npy"):
    # Cargamos la matriz
    Matrix = np.load(pathMatrix)
    # Cargamos las posiciones
    Position = np.load(pathPosition)
    media = np.mean(Matrix)
    std = np.std(Matrix)
    # Creamos el grafo
    grafoViaje = GrafoViaje(Matrix=Matrix,Position=Position,bordes=3,limite=media+(2*std))
    return grafoViaje
# Mostramos el grafoViaje
class Solucion:
    def __init__(self, pedidos : list = [], velocidad : int = 50, limiteTiempo : int = 8*3600, Capacidad : int = 15):
        # Solucion a problema de VRP
        self.Rutas = GenerarGrafo()
        self.Rutas.ReduceGraph()
        self.Acumulado = 0
        # Pedidos son ["Peso", "Nodo"]
        self.Pedidos = pedidos
        self.Velocidad = velocidad
        self.LimiteTiempo = limiteTiempo
        self.Capacidad = Capacidad
    
    def Equivalente(self, nodo):
        # Buscar en el nodo si contiene ese nombre
        for i in self.Rutas.Graph.nodes():
            for j in str(i).split(","):
                if j == str(nodo):
                    return i
        return nodo
    
    # Funcion que acumula los pedidos
    def AcumularPedidos(self):
        # Esta funcion junta todos los pedidos en un solo nodo al que pertenecen
        # Para esto, vamos a crear un diccionario, va a lo largo de los pedidos y los va a ir sumando
        Acumulados = {}
        for i in self.Pedidos:
            if self.Equivalente(i[1]) in Acumulados:
                Acumulados[self.Equivalente(i[1])] += i[0]
            else:
                Acumulados[self.Equivalente(i[1])] = i[0]
        # Ahora, vamos a crear una lista de pedidos
        self.Pedidos = []
        for i in Acumulados:
            self.Pedidos.append([Acumulados[i], i])
        # Ordenamos los pedidos
        self.Pedidos.sort(reverse=True)
        pass

    def Solve(self):
        # Funcion que resuelve el problema
        # PAra hacerlo creamos unas soluciones
        # Creamos una lista de soluciones que cubren todos los pedidos
        Soluciones = []
        # Creamos una lista de pedidos que no se han cubierto
        Pedidos = self.Pedidos
        # Ordenamos los pedidos de mayor a menor
        Pedidos.sort(reverse=True)
        # Creamos una lista de pedidos que se han cubierto
        Cubiertos = []
        # Empezamos a hacer una solucion hasta que ya no haya pedidos
        while len(Pedidos) > 0:
            print ("Longitud de pedidos", len(Pedidos))
            # Creamos una solucion
            Solucion = [0]
            # Capacidad de la solucion
            Capacidad = 0
            # Tiempo de la solucion
            Tiempo = 0
            # Limpieza de cubiertos
            Cubiertos = []
            # Checamos cuales son los nodos que se pueden cubrir
            for i in Pedidos:
                i = i
                # Checamos el nodo mas cercano y lo cambiamos a ese
                #for j in Pedidos:
                #    if self.Rutas.Graph.has_edge(Solucion[-1], self.Equivalente(j[1])):
                #        if self.calcTiempo(Solucion[-1], self.Equivalente(j[1])) < self.calcTiempo(Solucion[-1], self.Equivalente(i[1])):
                #            i = j
                        
                # Verificamos que no cubra mas de la capacidad
                #print ("Capacidad", Capacidad)
                if Capacidad + i[0] <= self.Capacidad and i not in Cubiertos and i[0] != 0:
                    #print ("Cubiertos", Cubiertos)
                    #print ("Cubiertos", Cubiertos)
                    # Verificamos que haya un camino entre los nodos
                    if self.Rutas.Graph.has_edge(Solucion[-1], self.Equivalente(i[1])):
                        #print ("Hay camino de ", Solucion[-1], "a", self.Equivalente(i[1]))
                        # Verificamos que no se pase del tiempo
                        if Tiempo + self.calcTiempo(Solucion[-1], self.Equivalente(i[1])) + self.calcTiempo(self.Equivalente(i[1]), 0) <= self.LimiteTiempo:
                            # Agregamos el nodo
                            Solucion.append(self.Equivalente(i[1]))
                            # Agregamos el peso
                            Capacidad += i[0]
                            # Agregamos el tiempo
                            Tiempo += self.calcTiempo(Solucion[-2], Solucion[-1])
                            # Agregamos el pedido a los cubiertos
                            Cubiertos.append(i)
                    #else:
                        #print ("No hay camino de ", Solucion[-1], "a", self.Equivalente(i[1]))
                        #print ("No hay camino de ", Solucion[-1], "a", i[1])
                
                # Verificamos si cubre la capacidad
                if Capacidad >= self.Capacidad:
                    break
            # Agregamos la solucion
            Soluciones.append(Solucion)
            time.sleep(0.1)
            # Eliminamos los pedidos cubiertos
            for i in Cubiertos:
                if i != 0:
                    Pedidos.remove(i)
            # Limpiamos los cubiertos
            Cubiertos = []
        # Regresamos las soluciones
        return Soluciones
    
    def calcTiempo(self, nodo1, nodo2):
        # Calculamos el tiempo entre dos nodos en segundos
        try:
            return (self.Rutas.Graph[nodo1][nodo2]['weight'] / self.Velocidad) * 3600
        except:
            #print ("No hay camino de ", nodo1, "a", nodo2)
            return 0

    def ShowGraph(self, name="Grafo.png"):
        self.Rutas.ShowRealGraph(name=name)
        pass

    def VisualizePath(self,path,name="Grafo.png"):
        # Creamos un nuevo grafo, nada mas actvamos las aristas que se usaron
        # Creamos el grafo
        grafoViaje = nx.Graph()
        # Agregamos los nodos
        grafoViaje.add_nodes_from(self.Rutas.Graph.nodes())
        # Agregamos las aristas
        for i in range(len(path)-1):
            grafoViaje.add_edge(path[i], path[i+1], weight=self.Rutas.Graph[path[i]][path[i+1]]['weight'])
        # Posicion
        position = dict(zip(grafoViaje.nodes(), self.Rutas.Position))
        # Dibujamos el grafo
        nx.draw(grafoViaje, position, with_labels=True, width=.3, node_size=80, font_size=5, font_color='black', node_color='pink', edge_color='black', alpha=0.9)
        # Guardamos la imagen
        plt.savefig(name, dpi=1000)
        plt.show()

    

if __name__ == "__main__":
    # Cargamos los datos
    Datos = Cargar_Datos()
    # Usamos la funcion de generar pedido
    numeroPedidos = 199
    pedidos = []
    for i in range(numeroPedidos):
        # Formato ["Peso", "Nodo"]
        pedidos.append([Generar_Pedido(Datos), i])
    # Print longitud de pedidos
    print("Longitud de pedidos original", len(pedidos))
    # Creamos la solucion
    solucion = Solucion(pedidos)
    # Acumulamos los pedidos
    solucion.AcumularPedidos()
    # Visualizamos el grafo
    solucion.ShowGraph()
    # Imprime las aristas
    print(solucion.Rutas.Graph.edges())
    # Imprimimos los pedidos
    #print(solucion.Pedidos)
    # Longitud de pedidos
    print("Longitud de pedidos acumulados", len(solucion.Pedidos))
    # Pedido mas grande
    print("Pedido mas grande", max(solucion.Pedidos))
    # Resolvemos el problema
    soluciones = solucion.Solve()
    # Imprimimos el tiempo total de cada solucion
    for i in soluciones:
        tiempo = 0
        for j in range(len(i)-1):
            tiempo += solucion.calcTiempo(i[j], i[j+1])
        print("Tiempo", tiempo)
        print("Tiempo en horas", tiempo/3600)
    # Imprimimos las soluciones
    print(soluciones)
    # Imprime la longitud de las soluciones
    print("Cantidad de carros", len(soluciones))
    # Visualizamos una solucion
    # Primero borramos el contenido de la carpeta Soluciones con os.system("rm -r Soluciones/*")
    for file in os.listdir("Soluciones"):
        os.remove("Soluciones/"+file)
    for solution in soluciones:
        solucion.VisualizePath(solution, name="Solucion"+str(soluciones.index(solution))+".png")
        # Guardamos la imagen en la carpeta Soluciones
        plt.savefig("Soluciones/Solucion"+str(soluciones.index(solution))+".png")
