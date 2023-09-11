from networkx import DiGraph
from vrpy import VehicleRoutingProblem
G = DiGraph()
G.add_edge("Source", 1, cost=1)
G.add_edge("Source", 2, cost=2)
G.add_edge(1, "Sink", cost=0)
G.add_edge(2, "Sink", cost=2)
G.add_edge(1, 2, cost=1)
G.nodes[1]["demand"] = 2
G.nodes[2]["demand"] = 3
prob = VehicleRoutingProblem(G, load_capacity=10)
prob.solve()
# Imprimir la solucion
print (prob.best_routes)