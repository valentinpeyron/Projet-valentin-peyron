
import networkx as nx
import matplotlib.pyplot as plt
import functions_part_2 as hh
import time

# :count the difference of ellement between two set ex (1,2) and (1,3) as one and (1), (1, 2) also one
def count_set_differences(set1, set2):
   
    diff_count = len(set1.symmetric_difference(set2))
    
    size_diff = abs(len(set1) - len(set2))
    
    total_diff = diff_count + size_diff
    
    return total_diff 


### test and plot  

l = int(input("test?(0,1,2,3): "))# 0 for toy exemple comparison, 1 for sub optimal grapg, 2 for graph of comparison between the algo

#this test the toy exemple 
if l==0:
    edges_U = [("z","t"),("t","s"),("w","t"),]
    edges_B = [("z","t"),("z","s"),("w","t"), ("w","s"),("t","z"),("s","z"),("t","w"), ("s","w")]#,("w","z"),("z","w")0
    G_U = nx.DiGraph(edges_U)
    G_B = nx.DiGraph(edges_B)
    S = {"s"}
    C = {"z":2,"w":2, "t":3}
    HH,flow_value, flow_dict, pas = hh.heuristic_algo(G_U,G_B,S,C)
    set_algo_1= hh.min_cost_intervention(S,G_U,G_B,C)
    print(set_algo_1)
    print( flow_dict)
    print(set_algo_1==flow_dict)

    plt.subplot(121)
    nx.draw(G_U, with_labels=True)
    plt.title('Graph Undirected')

    # Draw the second graph
    plt.subplot(122)
    nx.draw(G_B, with_labels=True)
    plt.title('Graph Bidirected')

    # Adjust the layout and display the graphs
    plt.tight_layout()
    plt.show()


#this part test to found a sub-optimal heuristic graph/ note that we need the while true boucle for the heuristic code to work. 
if l ==1:
    n = int(input("Nombre de sommets : "))
    ty = True
    flow_dict = set()#he.heuristic_algo(G_U,G_B,S,C)
    set_algo_1 =set()
    while ty:

        while True: 
            G_U = hh.generer_graphe_aleatoire(n,0)
            G_B = hh.generer_graphe_aleatoire(n,1)
            S = hh.generer_S(n, 1)#type 0 pour avoir un seul s
            print("S:"),print(S)
            
            C = hh.dictionnaire_poids(n,0,50)#prend en compte que si l'élémen est dans S il est infini
            flow_dict = hh.heuristic_algo(G_U,G_B,S,C)
            if len(flow_dict)!=0:
                break

        set_algo_1= hh.min_cost_intervention(S,G_U,G_B,C)
        ty = (set_algo_1==flow_dict)

    print(set_algo_1==flow_dict)
    print(set_algo_1)
    print(flow_dict)
    plt.subplot(121)
    nx.draw(G_U, with_labels=True)
    plt.title('Graph Undirected')

    plt.subplot(122)
    nx.draw(G_B, with_labels=True)
    plt.title('Graph Bidirected')

    plt.tight_layout()
    plt.show()
    
#this part give us the figure to compare the algorithm
if l ==2:
    nb_inf = int(input("numéro inf"))
    nb_sup = int(input("numéro sup"))
    range = range(nb_inf, nb_sup+1)
    temps_h = []
    temps_1 = []
    diff= []
    nodes = list(range)
    for i in range:
        n = i
        
        temps_debut_h = 0
        temps_fin_h = 0
        nb_1 = 0
        nb_h = 0
        cut_heuristic = set()
        #on utilise le while pour gèrer les graph posant problème 
        while True: 
            G_U = hh.generer_graphe_aleatoire(n,0)
            G_B = hh.generer_graphe_aleatoire(n,1)
            S = hh.generer_S(n, 0)#type 0 pour avoir un seul s
            C = hh.dictionnaire_poids(n,0,50)#prend en compte que si l'élémen est dans S il est infini
            temps_debut_h = time.time()
            cut_heuristic = hh.heuristic_algo(G_U,G_B,S,C)
            temps_fin_h = time.time()
            if len(cut_heuristic)!=0:break
        print("s")
        print(S)
        temps_debut_1 = time.time()
        cut_algo_1= hh.min_cost_intervention(S,G_U,G_B,C)
        temps_fin_1 = time.time()
        diff_h_1 = abs(count_set_differences(cut_heuristic, cut_algo_1))
        run_time_1= temps_fin_1 - temps_debut_1
        run_time_h = temps_fin_h - temps_debut_h
        temps_h.append(run_time_h)
        temps_1.append(run_time_1)
        diff.append(diff_h_1)


    # Tracer le graphique pour le run time 
    plt.plot(nodes, temps_h, marker='s', markersize=8, linestyle='-', color='blue', label='Algo_heuristics')
    plt.plot(nodes, temps_1, 'r-', label='Algo_1')
    plt.xlabel('Number of nodes')
    plt.ylabel('Run Time [s]')
    plt.title('Run time of both algorithm in fonction of the time')
    plt.legend()
    plt.grid(True)
    plt.show()
    # graphique de comparaison de justesse des resultat, on va utiliser le nb de nodes de différence comme metric
    plt.plot(nodes, diff, 'r-', label='Nodes difference')
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of differents nodes')
    plt.title('Comparative accuracy of Algo.1 versus Heuristc')
    plt.legend()
    plt.grid(True)
    plt.show()

# test pour algo 1 and heuristic (using heuristic_plus) using type = 1 you tes the case where S as multiple nodes
if l ==3:
    n = int(input("Nombre de sommets : "))
    type = 0 #0 for one nodes in S and 1 for multiple nodes in S
    set_algo_1 =set()
    flow_dict, G_B, G_U, S, C = hh.heuristis_plus(n, type)#this create directed and undirected graph and cost and S such that heuristic does not return "inf"
    #if you have directly the graph the cost and so on you need to only use heuristic_algo but not that heuristi_algo will return an empty set if either min_cut: nx.NetworkXUnbounded or pa_S_H is empty
    print("s")
    print(S)
    
    set_algo_1= hh.min_cost_intervention(S,G_U,G_B,C)
    
    print(set_algo_1)
    print(flow_dict)
    print(set_algo_1==flow_dict)



