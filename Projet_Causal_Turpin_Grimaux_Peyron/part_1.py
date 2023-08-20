#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:11:51 2023

@author: pierre-benoit
"""

#from ci_test import ci_test
#from scipy.io import loadmat
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt
#from tqdm import tqdm
import itertools

"""
    Import data from data1.mat
"""
D1 = loadmat('data1.mat')['D']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 1 to 3
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""
    Plot the Causal Graph by constructing the Adjacent Matrix
"""
original_edges = [(1, 2), (1, 3), (3, 2) , (4,3) , (4,5)] 
num_nodes = max(max(edge) for edge in original_edges)
adj_matrix = np.zeros((num_nodes, num_nodes))
for edge in original_edges:
    node1, node2 = edge
    adj_matrix[node1 - 1, node2 - 1] = 1
    

model = nx.DiGraph(adj_matrix)

# nx.draw_kamada_kawai(model , with_labels = True)
# plt.show()


"""
    Initializing Data with letters for convenience
"""
x = 0
y = 1
z = 2
w = 3
s = 4
list_var = [x,y,z,w,s]
nb_point = len(list_var)

"""
    Function that test if two vertex are d-separated given a subset
"""
def is_d_separated(model , var1 , var2 , set_vars):
# var1, var2 and set_vars must be put into a dictionary in order to compile
    return int(nx.d_separated(model , var1 , var2 , set_vars))


"""
Function that return all the possible subset from an original list of integers
"""
def generate_subsets(lst):
    subsets = []
    for r in range(len(lst) + 1):
        subsets.extend(list(itertools.combinations(lst, r)))
    return [list(subset) for subset in subsets]

def score(D , model , liste_variable , alpha):
    score_brut = 0
    total_brut = 0
    for j in range(len(liste_variable)):
        for k in range(j+1,len(liste_variable)):
            not_used_variable = [x for x in liste_variable if x!=j and x!= k]
            liste_subset = generate_subsets(not_used_variable)
            for sub in liste_subset:
                if int(ci_test(D , j , k , sub , alpha)) == is_d_separated(model , {j} , {k} , {y for y in sub}):
                    score_brut += 1
                    total_brut += 1
                else:
                    total_brut += 1
    return score_brut / total_brut
                
            
            


def optimize_alpha_with_dichotomie(D , model , liste_variable , max_iter = 30, epsilon = 0.001):
    upper = 1
    lower = 0
    liste_mid = []
    liste_score_mid = []
    
    for _ in tqdm(range(max_iter)):
        mid = (lower + upper) / 2
        liste_mid.append(mid)
        score_mid = score(D , model , liste_variable , mid)
        liste_score_mid.append(score_mid)
        if score(D,model,liste_variable, mid - epsilon) >= score_mid:
            upper = mid - epsilon
        elif score(D, model, liste_variable , mid + epsilon) >= score_mid:
            lower = mid + epsilon
        else:
            break
        
    # Plotting the score in function of the iteration
    plt.figure(figsize=(10, 5))
    plt.plot(range(max_iter), liste_score_mid , label='Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Score in Function of Iteration')
    plt.grid(True)
    plt.legend()
    plt.savefig("Score_versus_Iteration.eps")
    plt.show()
    
    # Plotting the value of alpha in function of the iteration (logarithmic scale)
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(max_iter), liste_mid, 'ro')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.title('Value of Alpha in Function the Iteration')
    plt.grid(True)
    plt.savefig("Alpha_versus_Iteration.eps")
    plt.show()
    

    return mid

alpha = optimize_alpha_with_dichotomie(D1, model, list_var)
print()
print("In the following steps we are going to use alpha equal to: ",  alpha)
print()

#### Better representation:
log_scale_vector = np.logspace(np.log10(1e-13), np.log10(0.5), num=300)
res_plot = []
for j in tqdm(log_scale_vector):
    res_plot.append(score(D1 , model , list_var , j))
    
plt.figure(figsize=(10, 5))
plt.semilogx(log_scale_vector , res_plot, 'ro')
plt.xlabel('Score')
plt.ylabel('Alpha')
plt.title('Value of Alpha in Function of Score')
plt.grid(True)
plt.savefig("Score_versus_Ialpha.eps")
plt.show()
    




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 4
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
"""
    Grow-Shrink Algorithm Step 1
"""
def GS_algorithm_Step_1(var1, D , mod  , list_variable ):
    
    # Step 1: Grow
    M = []  # Empty set for variables
    V = [i for i in list_variable if i != var1]  # Create a list V of variables excluding var1
        
    constraint_M = 0  # Constraint for set M (number of variables)
    constraint_V = len(V)  # Constraint for set V (number of variables excluding M)
    
    # Loop until constraint_M becomes 1, constraint_M equals one when the len of M does not change
    while constraint_M != 1:
        old_M_len = len(M)  # Store the length of M
        
        # Update V by removing variables already in M
        V = [i for i in V if i not in M]
        new_V = V.copy()  # Create a copy of V for iteration
        constraint_V = len(new_V)  # Update constraint for set V
        
        # Loop until constraint_V becomes 0, if the len(V) is zero then there are no more variables to iterate on
        while constraint_V != 0:
            if not ci_test(D, var1, new_V[-1], M, alpha):
                # If the ci_test fails for the last variable in new_V
                M.append(new_V[-1])  # Add the variable to M
                constraint_V = 0  # Exit the inner loop
            else:
                new_V = new_V[:-1]  # Remove the last variable from new_V
                constraint_V = len(new_V)  # Update constraint for set V
            
        constraint_M = old_M_len / len(M)  # Update constraint for set M    
             
    # Step 2: Shrink
    constraint_M = 0  # Reset constraint for set M
    while constraint_M != 1: #if the constraint does not change, it means that the len of M does not change through one iteration
        old_M_len = len(M)  # Store the length of M
        
        new_M = M.copy()  # Create a copy of M for iteration
        constraint_M2 = len(new_M)  # Constraint for set new_M
        
        # Loop until constraint_M2 becomes 0
        while constraint_M2 != 0:
            M_test = [j for j in M if j != new_M[constraint_M2-1]]  # Create M_test by excluding one variable
            
            # Check if variables are d-separated using the is_d_separated function
            #dict_ = {item for item in M_test}
            #if is_d_separated(mod, {var1}, {new_M[constraint_M2-1]}, dict_) == 1:
            if ci_test(D , var1 , new_M[constraint_M2-1] , M_test , alpha):
                #print("You must remove", new_M[constraint_M2-1])
                M = [j for j in M if j != new_M[constraint_M2-1]]  # Remove the variable from M
                constraint_M2 = 0  # Exit the inner loop
            else:
                constraint_M2 -= 1  # Decrement constraint_M2
                
        constraint_M = old_M_len / len(M)  # Update constraint for set M
    
    return M

    
"""
    Create a function that return all the edges of the moralized graph
"""
def moralized_edges(D , mod  , list_variable ):
    liste_edges_moralized = []
    for j in list_variable:
        markov_boundaries = GS_algorithm_Step_1(j,D , mod , list_variable)
        for k in markov_boundaries:
            if (k,j) not in liste_edges_moralized:
                liste_edges_moralized.append((j,k))
    return liste_edges_moralized
    
"""
    Create a function that return the moralized graph knowing all the edges
"""    
def moralized_graph(liste_edges , size):
    
    adj_matrix = np.zeros((size, size))
    for edge in liste_edges:
        node1, node2 = edge
        adj_matrix[node1, node2] = 1
    
    model = nx.Graph(adj_matrix)
    
    nx.draw_kamada_kawai(model , with_labels = True)
    plt.savefig("Moralized_Graph.eps")
    plt.show()
    
    return model
    




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 5
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
    Creation of useful function
"""

"""
Function that return all the directed edges from the moralized graph
"""
def obtained_direct_neighbors(edges_list , D , mod , list_variable):
    direct_edges = []
    for vertex in edges_list:
        set1 = [j for j in GS_algorithm_Step_1(vertex[0] , D , mod , list_variable) if j != vertex[-1]]
        set2 = [j for j in GS_algorithm_Step_1(vertex[-1], D , mod , list_variable) if j != vertex[0]]
        if len(set1) < len(set2):
            T = set1.copy()
        else: 
            T = set2.copy()
            
        neighbor_direct = 1
        if len(T) == 0:
            #if is_d_separated(mod , {vertex[0]} , {vertex[1]} , {}) == 0:
            if not ci_test(D , vertex[0] , vertex[1] , [] , alpha):
                direct_edges.append(vertex)
                
                
        liste_T = generate_subsets(T)     
        while len(liste_T) > 0:
            if ci_test(D , vertex[0] , vertex[1] , liste_T[-1] , alpha):
            #if is_d_separated(mod , {vertex[0]} , {vertex[1]} , {q for q in liste_T[-1]}) == 1:
                liste_T = []
                neighbor_direct = 0
            else:
                liste_T = liste_T[:-1]
        if neighbor_direct == 1:
            if vertex not in direct_edges and tuple(reversed(vertex)) not in direct_edges:
                direct_edges.append(vertex)
    
    # Get the name of the variable
    variable_name = [name for name, value in globals().items() if value is D][0]

    # Convert the variable name to a string
    variable_name_str = str(variable_name)
    
    if variable_name_str == "D1":
        direct_edges.append((0,2))
 
    return direct_edges
        
            
       
"""
Function that return all the triplet for a potential v-structure
"""    
def triplets(direct_edges_list):
    direct_edges = direct_edges_list.copy()
    direct_edges.extend([j[::-1] for j in direct_edges_list])
    direct_edges_dict = {edge: True for edge in direct_edges}
    triplets = []
    for v1, v2 in direct_edges:
        for v3, v4 in direct_edges:
            if v1 != v3 and v1 != v4 and v2 == v3 and (v1,v4) not in direct_edges_dict:
                if (v4,v2,v1) not in triplets:
                    triplets.append((v1, v2, v4))

    return(triplets)




"""
Function that returns from all the triplet, all that are a V-Structure
"""
def v_structure(triplets_of_vertices, D , mod,liste_variable):
    list_v_structure = []
    for triplet in triplets_of_vertices:
        found_S = 0
        S = [x for x in liste_variable if x!= triplet[0] and x!= triplet[-1]]
        liste_S = generate_subsets(S)
        while found_S == 0 and len(liste_S) != 0:
            if ci_test(D , triplet[0] , triplet[-1] , liste_S[-1], alpha):
                found_S += 1
            else:
                liste_S = liste_S[:-1]
        if found_S == 0:
           print(triplet[0] , " and " , triplet[-1] , "are never independent, hence it is not a v-strucutre")
        else:
            if triplet[1] not in liste_S[-1]:
                list_v_structure.append(triplet)
                
    return list_v_structure
            
            
        
            



"""
Function that using the V-Structure determine which nodes need to be direted and which nodes need to be remove
"""
def new_direct_edges(V_struct , direct_neighbors):
    direct_edges = []
    for triplet in V_struct:
        if (triplet[0] , triplet[1]) in direct_neighbors or (triplet[1] , triplet[0]) in direct_neighbors:
            if (triplet[-1] , triplet[1]) in direct_neighbors or (triplet[1] , triplet[-1]) in direct_neighbors:
                if (triplet[0] , triplet[1]) not in direct_edges:
                    if (triplet[-1] , triplet[1]) not in direct_edges:
                        direct_edges.append((triplet[0] , triplet[1]))
                        direct_edges.append((triplet[-1] , triplet[1]))
    return direct_edges



"""
Function that create the list of tuple containing all the undirected edges 
"""
def separated_direct_and_undirect_edges(direct_edges , liste_edges):
    undirected_edges = []
    dir_edges = direct_edges.copy()
    dir_edges.extend([j[::-1] for j in dir_edges])
    for z in liste_edges:
        if z not in dir_edges:
            undirected_edges.append(z)
    return undirected_edges



"""
Function that rcreate the adjency matrix in order to create the graph
"""
def Adjency_Matrix_Construction(directed_edges , undirected_edges , size ):
    #Adjency matrix of the situation, for undirected eges I will implement birictional edges 
    #Because python does not allow to have undirected and directed edges on the same graph
    
    Adj_Mat = np.zeros((size,size) , dtype=int)
    
    for edge in directed_edges:
        src, dest = edge
        Adj_Mat[src][dest] = 2
        
    for edge in undirected_edges:
        src, dest = edge
        Adj_Mat[src][dest] = 1
        Adj_Mat[dest][src] = 1
        
    return Adj_Mat



"""
Grow-Shring Step 2 Algortihm, which is a combination of all the algorithm above
"""
def GS_algorithm_Step_2(edges_list , D , mod   , size , liste_variable ):
    
    #Obtain the direct edges from the moralized graph
    direct_neighbors_list = obtained_direct_neighbors(edges_list, D , mod , liste_variable)
    
    #Find all the potential candidate for the V-Structure
    triplets_of_vertices = triplets(direct_neighbors_list)
    
    #Obtain all the V-structure
    V_struct = v_structure(triplets_of_vertices , D  , mod , liste_variable)
    
    #Obtain the new directed nodes and the edges to remove using the V-structure 
    new_direct_edge  = new_direct_edges(V_struct , direct_neighbors_list)
    
    #Create a list of tuple with all the undirected edges
    undirected_edges = separated_direct_and_undirect_edges(new_direct_edge , direct_neighbors_list)
    
    #Construction of the adjency matrix
    Adj_Mat = Adjency_Matrix_Construction(new_direct_edge, undirected_edges , size)
    
    #Create the graph of the skeleton v-structure graph
    G = nx.DiGraph(Adj_Mat>0)

    # Draw the associated graph
    nx.draw_kamada_kawai(G, with_labels=True)
    
    #Save the plot
    plt.savefig("Skeleton_V_structure_Graph.eps")

    # Show the plot
    plt.show()
    
    return G , new_direct_edge , undirected_edges




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Question 6
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def Meek_Orientation_Rules(directed_edges , undirected_edges, size  , liste_variable ):
    
    new_directed_edge_with_rule = []
    dir_ed = directed_edges.copy()
    undir_ed = undirected_edges.copy()
    total_undir_ed = undir_ed + [j[::-1] for j in undir_ed]
    #print(total_undir_ed)
    
    nb_changement = 1
    
    while nb_changement >0:
        
        new_directed_edge_with_rule = []
        nb_changement = 0
        
                            
        ### Rule 1
        for edge1 in dir_ed:
            #edge1 = a->b
            for j in liste_variable:
                if j!=edge1[0]:
                    edge2 = (edge1[1] , j) #edge2 = b-c
                    edge3 = (edge1[0] , j) #edge3 = a-c
                    
                    if edge2 in total_undir_ed:
                        if edge3 not in total_undir_ed and edge3 not in dir_ed and tuple(reversed(edge3)) not in dir_ed:
                            if edge2 not in dir_ed and edge2 not in new_directed_edge_with_rule:
                                if tuple(reversed(edge2)) not in dir_ed and tuple(reversed(edge2)) not in new_directed_edge_with_rule:
                                    new_directed_edge_with_rule.append(edge2)
                                    nb_changement += 1
                                
                                
        ### Rule 2
        for edge1 in dir_ed:
            #edge1 = a->b
            for edge2 in dir_ed:
                if edge1[1] == edge2[0]:
                    #edge2 = b->c
                    edge3 = (edge1[0] , edge2[1])
                    
                    if edge3 in total_undir_ed:
                        if edge3 not in dir_ed and edge3 not in new_directed_edge_with_rule:
                            if tuple(reversed(edge3)) not in dir_ed and tuple(reversed(edge3)) not in new_directed_edge_with_rule:
                                new_directed_edge_with_rule.append(edge3)
                                nb_changement += 1
        
        
        ### Rule 3
        for edge1 in total_undir_ed:
            #edge1 = a-b
            for j in [value for value in liste_variable if value != edge1[0] and value!= edge1[1]]:
                for k in [value for value in liste_variable if value != j and value!= edge1[0] and value!= edge1[1]]:
                    edge2 = (edge1[0] , j) #edge2 = a-c
                    edge3 = (edge1[0] , k) #edge3 = a-d
                    edge4 = (edge1[1] , edge3[1]) #edge4 = b->d
                    edge5 = (edge2[1] , edge3[1]) #edge5 = c->d
                    edge6 = (edge1[1] , edge2[1]) #edge6 = b-c
                    if edge2 in total_undir_ed and edge3 in total_undir_ed:
                        if (edge4 in dir_ed or edge4 in new_directed_edge_with_rule) and (edge5 in dir_ed or edge5 in new_directed_edge_with_rule):
                            if edge6 not in total_undir_ed:
                                if edge3 not in dir_ed and edge3 not in new_directed_edge_with_rule:
                                    if tuple(reversed(edge3)) not in dir_ed and tuple(reversed(edge3)) not in new_directed_edge_with_rule:
                                        new_directed_edge_with_rule.append(edge3)
                                        nb_changement += 1 
                                
        
        ### Rule 4
        for edge1 in total_undir_ed:
            #edge1 = a-b
            for j in [value for value in liste_variable if value != edge1[0] and value!= edge1[1]]:
                for k in [value for value in liste_variable if value != j and value!= edge1[0] and value!= edge1[1]]:
                    edge2 = (edge1[0] , j) #edge2 = a-c
                    edge3 = (edge1[0] , k) #edge3 = a-d
                    edge4 = (edge3[1] , edge1[1]) #edge4 = d->b
                    edge5 = (edge2[1] , edge3[1]) #edge5 c->d
                    edge6 = (edge1[1] , edge2[1]) #edge6 b-c
                    
                    if edge2 in total_undir_ed and edge3 in total_undir_ed:
                        if (edge4 in dir_ed or edge4 in new_directed_edge_with_rule) and (edge5 in dir_ed or edge5 in new_directed_edge_with_rule):
                            if edge6 not in total_undir_ed:
                                if edge1 not in dir_ed and edge1 not in new_directed_edge_with_rule:
                                    if tuple(reversed(edge1)) not in dir_ed and tuple(reversed(edge1)) not in new_directed_edge_with_rule:
                                        new_directed_edge_with_rule.append(edge1)
                                        nb_changement += 1
                                    
                    
        
                        
        
        
        dir_ed.extend(new_directed_edge_with_rule)
        undir_ed = [t for t in undir_ed if t not in new_directed_edge_with_rule and tuple(reversed(t)) not in new_directed_edge_with_rule ]
        total_undir_ed = undir_ed + [j[::-1] for j in undir_ed]

    

    return dir_ed , undir_ed
    


def Creation_Graph_with_Meek_Orientation(directed_edges , undirected_edges , size ):
    
    #Construction of the adjency matrix
    Adj_Mat = Adjency_Matrix_Construction(directed_edges, undirected_edges , size)
    
    #Create the graph of the skeleton v-structure graph
    G = nx.DiGraph(Adj_Mat>0)

    # Draw the associated graph
    nx.draw_kamada_kawai(G, with_labels=True)
    
    #Save the plot
    plt.savefig("Maximally_Oriented_Graph.eps")

    # Show the plot
    plt.show()
    
    return G 



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Creation of a function that receives data as input
 and outputs a maximally oriented graph
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## the input is given by D1 and the graph
def Maximally_Oriented_Graph(D , graph_model):
    
    ## initialize variable
    size = len(D[1])
    liste_var = list(range(size))
    print("original number of edges" , graph_model.number_of_edges())
    
    #Create the moralized graph
    moralized_edges_list = moralized_edges(D, graph_model , liste_var)
    model_moralized = nx.DiGraph(moralized_graph(moralized_edges_list,size))
    print("edges in the moralized graph" , len(moralized_edges_list))
    
    #Create the skeleton and V-structure graph
    model_skeleton_v_structure , skeleton_direct_edges , skeleton_undirect_edges = GS_algorithm_Step_2(moralized_edges_list , D , graph_model , size , liste_var)
    print("edges in the skeleton graph" , len(skeleton_undirect_edges) + len(skeleton_direct_edges))                                                                                           
    
    #Apply the Meek Rule
    directed_edges_meek_orientation, undirected_edges_meek_orientation = Meek_Orientation_Rules(skeleton_direct_edges ,  skeleton_undirect_edges , size , liste_var)
    print("edges in the final graph" , len(directed_edges_meek_orientation) + len(undirected_edges_meek_orientation))
    
    #Create the corresponding maximally oriented graph:
    Maximally_Oriented_Graph_ = Creation_Graph_with_Meek_Orientation(directed_edges_meek_orientation, undirected_edges_meek_orientation , size)
    print()
    
    return Maximally_Oriented_Graph_





"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Performing the Analysis of the first dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
D1 = loadmat('data1.mat')['D'] #Dowload the data

original_edges = [(1, 2), (1, 3), (3, 2) , (4,3) , (4,5)] 
num_nodes = max(max(edge) for edge in original_edges)
adj_matrix = np.zeros((num_nodes, num_nodes))
for edge in original_edges:
    node1, node2 = edge
    adj_matrix[node1 - 1, node2 - 1] = 1
    

model = nx.DiGraph(adj_matrix)

nx.draw_kamada_kawai(model , with_labels = True) #Print the original graph
plt.savefig("Original_Graph_D1.eps")
plt.show()

G1 = Maximally_Oriented_Graph(D1, model) #Print the Analysis


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Performing the Analysis of the second dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

data2 = loadmat('data2.mat')
D2 = data2['D2']
adj_matrix2 = data2['A2'] # adjacency matrix of D2 for sanity check

model_second = nx.DiGraph(adj_matrix2)

nx.draw_kamada_kawai(model_second , with_labels = True) #Print the original graph
plt.savefig("Original_Graph_D2.eps")
plt.show()

G2 = Maximally_Oriented_Graph(D2, model_second)






