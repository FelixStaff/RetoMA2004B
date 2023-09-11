# Importar el kmeans
from sklearn.cluster import KMeans
import numpy as np
import random
import matplotlib.pyplot as plt
# Ignorar los warnings
import warnings
warnings.filterwarnings("ignore")
from python_tsp.heuristics import solve_tsp_local_search
import networkx as nx
from tqdm import tqdm
import time
# Cargamos las coordenadas
class Clusters:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        # Cargar matriz de distancias
        self.matrixAd = np.load("matrixT.npy")
        # Cantidad de nodos
        self.nodos = len(self.matrixAd)
        # Cargamos los datos
        self.Coordenadas = np.load("coordenadas.npy")
        # Convertimos a un array
        self.Coordenadas = np.array([self.Coordenadas[:,0], self.Coordenadas[:,1]]).T
        # Aplicamos el kmeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.Coordenadas)
        # Obtenemos los centroides
        self.centroides = self.kmeans.cluster_centers_
        # Obtenemos las etiquetas
        self.etiquetas = self.kmeans.labels_
        # Creamos K subgrafos
        self.subgrafos = []
        # Valores de los clusters
        self.valores = []
        # Guardamos en subgrafos los nodos que pertenecen a cada cluster
        for i in range(n_clusters):
            self.subgrafos.append([])
            self.valores.append([])
        # Guardamos las coordenadas de cada cluster
        for i in range(len(self.etiquetas)):
            self.subgrafos[self.etiquetas[i]].append(self.Coordenadas[i])
            self.valores[self.etiquetas[i]].append(i)
    
    def graficar(self):
        # Graficamos los centroides
        for i in range(len(self.centroides)):
            plt.scatter(self.centroides[i][0], self.centroides[i][1], s=10, c='red')
            plt.annotate(i, (self.centroides[i][0], self.centroides[i][1]))
        # Graficamos los nodos
        for i in range(len(self.subgrafos)):
            subgrafo = np.array(self.subgrafos[i])
            plt.scatter(subgrafo[:,0], subgrafo[:,1], s=10)
        plt.show()

    def getSubgrafos(self):
        return self.subgrafos
    
    def getAdyacencia(self, n_cluster):
        # Creamos la matriz de adyacencia
        adyacencia = np.zeros((len(self.subgrafos[n_cluster]), len(self.subgrafos[n_cluster])))
        # Llenamos la matriz de adyacencia
        for i in range(len(self.subgrafos[n_cluster])):
            for j in range(len(self.subgrafos[n_cluster])):
                # Si es el mismo nodo, entonces es 0
                if i == j:
                    adyacencia[i][j] = 0
                else:
                    # Si no, entonces es la distancia entre ambos
                    # Lo multiplicamos por un numero random entre .95 y 1.05
                    randTime = random.uniform(.85, 1.35)
                    adyacencia[i][j] = self.matrixAd[self.valores[n_cluster][i]][self.valores[n_cluster][j]] * randTime
        return adyacencia
    
    def getClusterValores(self, n_cluster):
        return self.valores[n_cluster]

if __name__ == "__main__":
    # Colores para visualizar
    # Matriz de valores iteraciones
    NCamiones = 10
    NRampas = 10
    DatosRecogidos = np.zeros((NCamiones, NRampas,6))
    PromedioGlobal = 0
    # ITERACIONES
    ITERACIONES = 200 
    VISUALIZAR = False
    colores = ['red', 'blue', 'green', 'orange', 'pink', 'black', 'gray', 'purple', 'brown']
    # Creamos los clusters\
    for n in range(1, NCamiones+1):
        for m in range(1, NRampas+1):
            Entregas = []
            PromedioGlobal = 0
            PromedioEntregas = [0 for i in range(8)]
            barra = tqdm(range(ITERACIONES))
            for j in barra:
                clusters = Clusters(8)
                # Graficamos
                if VISUALIZAR == True:
                    clusters.graficar()
                promedio_distancias = 0
                for i in range(clusters.n_clusters):
                    matrizAdyacencia = clusters.getAdyacencia(i)
                    # Obtenemos los valores de cada cluster
                    valores = clusters.getClusterValores(i)
                    # Aplicamos el tsp
                    permutation, distance = solve_tsp_local_search(matrizAdyacencia)
                    #print (permutation, distance / 3600)
                    promedio_distancias += (distance) / 3600
                    # Agregamos el promedio de las distancias
                    PromedioEntregas[i] += (distance) / 3600
                    # Visualizamos el grafo
                    if VISUALIZAR == True:
                        
                        # Creamos un grafo
                        G = nx.Graph()
                        # Agregar los nodos
                        G.add_nodes_from(range(len(valores)))
                        # Agregar las aristas de la solucion
                        # Agregamos un nodo de 0 al primero
                        if valores[permutation[0]] != 0:
                            G.add_edge(0, valores[permutation[0]])
                        for j in range(len(permutation)-1):
                            if valores[permutation[j]] == valores[permutation[j+1]]:
                                continue
                            if valores[permutation[j]] == 0 and valores[permutation[j+1]] == 0:
                                continue
                            G.add_edge(valores[permutation[j]], valores[permutation[j+1]])
                        # Luego un nodo del ultimo al 0
                        if valores[permutation[-1]] != 0:
                            G.add_edge(valores[permutation[-1]], 0)
                        # Visualizar el grafo pero cambiando el color
                        nx.draw(G, clusters.Coordenadas, width=1.0, with_labels=True, node_size=50, font_size=5, font_color='black', node_color='pink', edge_color=colores[i], alpha=0.9, arrows=True)
                if VISUALIZAR == True:
                    plt.show()
                # print ("Promedio de distancias: ", promedio_distancias / clusters.n_clusters)
                PromedioGlobal += promedio_distancias / clusters.n_clusters
                # Calculamos el tiempo promedio de entrega
                barra.set_description("Numero de camiones: %d, Numero de rampas: %d, Promedio de distancias: %f" % (n, m, PromedioGlobal/(j+1)))
            LaMedia = PromedioGlobal / ITERACIONES
            # Calculamos las entregas
            for i in range(8):
                PromedioEntregas[i] = PromedioEntregas[i] / ITERACIONES
                Entregas.append(PromedioEntregas[i])
            Lambda = n/LaMedia
            import random
            import numpy as np
            # Generamos 199 puntos aleatorios con distribución de poisson
            promedio = 0
            for i in range(1000):
                lam = 1.41
                K = 199
                salidas = [] # Guardamos los numeros aleatorios de poisson
                for i in range(K):
                    salidas.append(np.random.poisson(lam))
                # Dividimos en 5 grupos
                grupos = []
                for i in range(5):
                    grupos.append(salidas[i*40:(i+1)*40])
                # Vemos cuanto suman en promedio los grupos
                suma = 0
                for i in range(5):
                    suma += sum(grupos[i])
                promedio += suma/5
            promedio = promedio/1000 
            tiempoCarga = promedio*.35 / 60 * n/m
            tiempoAlistar = promedio*.15 / 60 * n/m
            #print ("Tiempo de alistamiento: ", tiempoAlistar)
            #print ("Tiempo de carga: ", tiempoCarga)
            mu = 60/tiempoCarga * m
            # Imprimir las entregas
            print ("Entregas: ", Entregas)
            #print ("Tiempo de servicio: ", tiempoCarga)
            #print ("Mu: ", mu)
            #print ("Lambda: ", Lambda)
            #print ("Taza de utilizacion: ", Lambda/mu)
            # Guardamos el promedio global
            DatosRecogidos[n-1][m-1][0] = mu
            DatosRecogidos[n-1][m-1][1] = Lambda
            DatosRecogidos[n-1][m-1][2] = Lambda/mu
            DatosRecogidos[n-1][m-1][3] = LaMedia
            # Guardamos el archivo
            np.save("DatosRecogidos.npy", DatosRecogidos)
            # Analizamos cuanto seria el tiempo entre llegadas
            # El maximo es 8 horas, asi que si supera ese tiempo, lo pasamos a otro y le agregamos 
            # Analizamos cuanto seria el tiempo entre llegadas
            # El maximo es 8 horas, asi que si supera ese tiempo, lo pasamos a otro y le agregamos el tiempo que sobra al siguiente
            Camiones = [{'Tiempo' : tiempoAlistar, "Dias" : 0} for i in range(n)]
            print ("Numero de entregas: ", len(Entregas))
            minDia = 0
            while len(Entregas) > 0:
                # Vamos de camion en camion
                for i in range(n):
                    # Vamos de tarea en tarea
                    TareasCompletadas =[]
                    if Camiones[i]['Dias'] > minDia:
                        
                        continue
                    for j in range(len(Entregas)):
                        # Si el tiempo es menor a 8 horas, entonces lo agregamos
                        if Camiones[i]['Tiempo'] + Entregas[j] + tiempoCarga <= 8:
                            Camiones[i]['Tiempo'] += Entregas[j] + tiempoCarga
                            TareasCompletadas.append(Entregas[j])
                        else:
                            # Si es mayor, cortamos la tarea y la agregamos como si fuera una nueva
                            Entregas.append(Entregas[j] - (8 - (Camiones[i]['Tiempo'] + tiempoCarga)))
                            Camiones[i]['Tiempo'] = 0
                            Camiones[i]['Tiempo'] += tiempoAlistar
                            Camiones[i]['Dias'] += 1
                            TareasCompletadas.append(Entregas[j])
                            break
                    # Eliminamos las tareas completadas
                    for j in range(len(TareasCompletadas)):
                        Entregas.remove(TareasCompletadas[j])
                    # Si ya no hay tareas, entonces salimos
                    if len(Entregas) == 0:
                        break
                    # Si no, entonces pasamos al siguiente camion
                # minDia es el minimo de todos los dias de cada camion
                minDia = min([Camiones[i]['Dias'] for i in range(n)])
                # Si ya no hay tareas, entonces salimos
                if len(Entregas) == 0:
                    break
            # Calculamos el tiempo promedio de entrega
            suma = 0
            for i in range(n):
                suma += Camiones[i]['Dias'] * 8 + Camiones[i]['Tiempo']
            suma = suma/n
            # print ("Tiempo promedio de entrega: ", suma)
            DatosRecogidos[n-1][m-1][4] = suma / 8
            print ("Tiempo promedio de entrega (dias): ", suma/8)
            np.save("DatosRecogidos.npy", DatosRecogidos)