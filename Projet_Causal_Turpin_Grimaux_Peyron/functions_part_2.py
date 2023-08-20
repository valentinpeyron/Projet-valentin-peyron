import networkx as nx
import random
from itertools import combinations
import sys
import numpy as np
import matplotlib.pyplot as plt


## Question 0 : generate random graphs and weights 

#create a dictionary with the cost of each element of the grapg not in S
def dictionnaire_poids(taille, limite_b, limite_h):

    dictionnaire = {}

    for i in range(taille):
        valeur = random.randint(limite_b, limite_h) 
        dictionnaire[i] = valeur

    return dictionnaire

def generer_S(n_vertices, type):
    S = set()
    if type == 0:
           S.add(random.randint(1, n_vertices-1))
    else:
        for v1 in range(n_vertices):
                v = random.randint(1, n_vertices-1)
                if v1 ==1:  S.add(v)
                if random.random() < 0.2 and v1!=v:  # Probabilité de 0.5 pour avoir une arête entre deux sommets
                    S.add(v1)
    return S

def generer_graphe_aleatoire(n_vertices, type): #type et 0 pour unidirectionnelle et 1 pour bidirectionnelle
    G = nx.DiGraph()  # Création d'un graphe vide
    # Ajouter les sommets au graphe
    for v in range(n_vertices):
        G.add_node(v)
    # Ajouter des arêtes aléatoires entre les sommets
    for v1 in range(n_vertices):
        for v2 in range(v1 + 1, n_vertices):
            if random.random() < 0.5:  # Probabilité de 0.5 pour avoir une arête entre deux sommets
                if type == 0: G.add_edge(v1, v2)
                else:
                    G.add_edge(v1, v2)
                    G.add_edge(v2, v1)
    return G

## Question 1 : Implement the function H_hull

def find_maximal_c_component_containing_S(S, graph):
    max_cc = list(nx.connected_components(graph))
    for c in max_cc :
        if c & S:     # At least one element in common between c and S
            return c  # Assuming again only one c comp

def find_ancestors_S(S,graph):
    # Creates a set of all ancestors of S
    ancestors=[]
    for s in S:
        ancestors += nx.ancestors(graph, s)  ### Probleme a trouver ancestors si s pas dans graph
    ancestors = set(ancestors)
    return ancestors       
 

def H_hull(S,G1,G2):
    # G1: DiGraph uni arrow
    # G2: DiGraph bi arrowa 
    G2 = nx.Graph(G2)  # Undirected graph
    F = set(G1.nodes())
    
    #special case
    if not S.issubset(F):
        sys.exit("h_Hull: S is not in the graph")

    while True:
        F1 = find_maximal_c_component_containing_S(S,G2.subgraph(F))  ### Asuming only one c-comp
        S = S & F1
        F2 = find_ancestors_S(S,G1.subgraph(F1))| S      
        if F2 != F:
            F=F2
        else:
                return F
            
## Question 2 : weighted minimum hitting set of these sets

def min_cost_hitting_set(sets, costs):
    # convert lists to sets
    sets = [set(subset) for subset in sets]
    universe = set.union(*sets)
    all_subsets = []
    for r in range(1, len(universe) + 1):
        for subset in combinations(universe, r):
            all_subsets.append(set(subset))
    hitting_sets = [s for s in all_subsets if all(any(e in s for e in subset) for subset in sets)]
    return min(hitting_sets, key=lambda s: sum(costs[e] for e in s))

## Quesion 3 : Implement algorithm 1
            
def min_cost_intervention_one_cc(S,G1,G2,weight):
    # S:  set
    # G1: unidirected graph
    # G2: bidirected graph
    # weight : dictionnary of the type weight={1:3,2:2,3:2} 
    V = set(G1.nodes())
    F=[] #array of array
    H=H_hull(S,G1,G2)

    if not H-S: 
        return {}
    while True:
        while True:
            filtered_weight = {k: v for k, v in weight.items() if k in H-S}
            a = min(filtered_weight, key=weight.get)
            a = set([a])
        
            if not H_hull(S,G1.subgraph(H-a),G2.subgraph(H-a)) - S :
                F.append(H)
                break
            else :
                H = H_hull(S,G1.subgraph(H-a),G2.subgraph(H-a))
            
        set_of_sets=set_of_sets_WMHS(F,S,G2)
        A = min_cost_hitting_set(set_of_sets,weight)
        if not H_hull(S,G1.subgraph(V-A),G2.subgraph(V-A)) - S :
                return A
        H = H_hull(S,G1.subgraph(V-A),G2.subgraph(V-A))
        
##Question 4: Implement heuristics

#this algo create the grap the cost and the S adaptated for heuristics to work and then return them for algo_1 to use them.
def heuristis_plus(n,type):
     while True: 
            G_U = generer_graphe_aleatoire(n,0)#\graphe unidirectionnelle
            G_B = generer_graphe_aleatoire(n,1)#graphe 
            S = generer_S(n, type)#type 0 pour avoir un seul s
            #print("S:"),print(S)
            
            C = dictionnaire_poids(n,0,50)#prend en compte que si l'élémen est dans S il est infini
            flow_dict = heuristic_algo(G_U,G_B,S,C)
            if len(flow_dict)!=0:
                return flow_dict, G_B, G_U, S, C
            

def heuristic_algo(G_U,G_B,S,cost): 
    H = H_hull(S,G_U,G_B)

    new_costs = {key: value for key, value in cost.items() if key not in S}
    pa_S = set()
    for v in S:
        predecessors = G_U.predecessors(v)
        pa_S.update(predecessors)
    pa_S_H = pa_S.intersection(H)
    

    Sxy = {"x", "y"} | S
    # calcul of min-flow by cretating a new double graph and verifying that pa_s_H is not empty. because this case is useless and not working
    if len(pa_S_H)!=0: cut_elements = Min_cut_graph (G_B.subgraph(H), pa_S_H, S, new_costs, Sxy)
    else: 
        #print("erreur")
        return set()
    if cut_elements=="inf": return set()#this allows to launch again in order to get
    else:
        #print(cut_elements)
        cut_elements = [elem.split("_", 1)[0] for elem in cut_elements]
        #print(cut_elements)
        cut_elements = [int(elem) for elem in cut_elements]
        cut_elements= {v for v in cut_elements if cut_elements.count(v) == 1}
        return  cut_elements 


#this function is creating the weighted grapg by doubling the vertex not in S
def Min_cut_graph (graph, pa_interH, S, cost, Sxy):
    type = nx.is_directed(graph)#return if g id directed or not

    # we verify that S, x,y are not in min-cut:
    for s in Sxy:
        cost[s] = np.inf
    # transform vertex-cut to edge-cut:
    HH = nx.DiGraph()
    for v in graph.nodes:
        HH.add_edge(str(v) + "_in", str(v) + "_out", capacity=cost[v])
        for w in graph.adj[v]:  # sucessors  of  v in the  graph:
            HH.add_edge(str(v) + "_out", str(w) + "_in", capacity=np.inf)
    
    for p in pa_interH:
        HH.add_edge("x", str(p) + "_in", capacity=np.inf)
        if not type:
            HH.add_edge(str(p) + "_out", "x", capacity=np.inf)
    
    for s in S:
        HH.add_edge(str(s) + "_out", "y", capacity=np.inf)
        if not type:
            HH.add_edge("y_target", str(t) + "/1", capacity=np.inf)

    # we apply min_cut:
    try:
        cut_value, (cut_noeud, remaining_noeud) = nx.minimum_cut(HH, "x", "y")
    except nx.NetworkXUnbounded:
        return 'inf'

    # the cut is the smallest of cut_noeud, remaining_noeud:
    if len(cut_noeud) < len(remaining_noeud):
        retour = list(cut_noeud)
        retour.remove("x")
    else:
        retour = list(remaining_noeud)
        retour.remove("y")

    # we verifu the repetition
    node_list = [t.split('/')[0] for t in retour]
    return [v for v in node_list if node_list.count(v) == 1]

## Question 5 : G[S] consists of several maximal c-components

def min_cost_intervention(S,G1,G2,weight):
    # S:  set
    # G1: unidirected graph
    # G2: bidirected graph
    # weight : dictionnary of the type weight={1:3,2:2,3:2} 
    V = set(G1.nodes())
    F=[] #array of array
 
    min_cost=[]
    partitionning_S = find_subsets_S_cc(S, G2)
    
    for s in partitionning_S:
        H=H_hull(s,G1,G2)
    
        if not H-s: 
            min_cost.append({})
            break
        while True:
            while True:
                filtered_weight = {k: v for k, v in weight.items() if k in H-s}
                a = min(filtered_weight, key=weight.get)
                a = set([a])
                
                if not H_hull(s,G1.subgraph(H-a),G2.subgraph(H-a)) - s :
                    F.append(H)
                    break
                else :
                    H = H_hull(s,G1.subgraph(H-a),G2.subgraph(H-a))
          
            set_of_sets=set_of_sets_WMHS(F,s,G2)
            A = min_cost_hitting_set(set_of_sets,weight)
            if not H_hull(s,G1.subgraph(V-A),G2.subgraph(V-A)) - s :
                    min_cost.append(A)
                    break
            H = H_hull(s,G1.subgraph(V-A),G2.subgraph(V-A))
    return set().union(*min_cost)
        
def find_subsets_S_cc(S, graph):
    # Find all the partitionning of G_[S] into its maximal c_components
    graph=nx.Graph(graph) #undirected graph
    max_cc = list(nx.connected_components(graph))
    subsets = []
    
    for c in max_cc:
        if c & S: # At least one element in common between c and S
            subsets.append(c & S)
    return subsets  
                
def set_of_sets_WMHS(F,S,graph):
    # F : set of sets
    # S : set
    subsets_S_cc = find_subsets_S_cc(S, graph)
    # Generate all possible combinations of differences
    combinations = [f - s for f in F for s in subsets_S_cc]
    return(combinations)

        
        
                
                
            
                
                