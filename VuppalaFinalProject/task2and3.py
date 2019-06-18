#PYTHON 3.6
import networkx as nx 
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from statistics import mean, stdev, median
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
#Adds an asterisk at the start
def asterisk_commentor(node):
    node1 = '@' + node
    return node1
#Removes an asterisk at the start
def removeAsterisk(node):
    return node.replace('@', '')

#Adds an edge to the graph only if the edge doesn't already exist, otherwise increases the edge weight
def addEdge(G, a, b, role1):
    if G.has_edge(a, b):
        G[a][b]['weight'] += 1
    else:
        G.add_edge(a, b, weight = 1, role = role1)
    return G
#Displays mean and standard deviation
def printMeanAndSD(data):
    print('Mean: ', mean(data))
    print('Standard Deviation: ', stdev(data))

def calculateMeanMedSD(data):
    m = mean(data)
    med = median(data)
    sd = stdev(data)
    return m, med, sd
        
#Calculates the coefficients of the regression line and also gives the predicted y values using sklearn
def lr(Xtest, Ytest):
    X1 = np.asarray(Xtest).reshape(-1, 1)
    regr = LinearRegression()
    regr.fit(X1, Ytest)
    YPred = regr.predict(X1)
    return YPred, regr.coef_

#Returns true if the edge role of the bridge or local bridge is a bully
def IsBridgeBetweenBullyAndVictim(graph1, b):
    if (graph1[b[0]][b[1]]['role'] == 'bully' or graph1[b[0]][b[1]]['role'] == 'defender;bully' or
        graph1[b[0]][b[1]]['role'] == 'assistant;bully'):
        return True
    else:
        return False
#Calculates the principal eigen values of  a graph
def eigenvalues(mat):
    tmp = nx.adjacency_matrix(mat, nodelist=sorted(mat.nodes()))
    tmp = tmp.toarray()
    w, v = np.linalg.eig(tmp)
    return w, v

#Plots the graphs
def plotGraph(Xs, Ys, l1, l2, t1):
    if Xs == Ys:
        print(len(Xs), len(Ys))
    fig = plt.figure(1)
    plt.scatter(Xs, Ys)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(l1)
    plt.ylabel(l2)
    plt.title(t1)
    plt.show()
    

#Load the csv files data into a dataframe
data = pd.read_csv('dataset_5.csv', sep=',', header=0)

X = data.values
#print(X)
#print(X.shape)

###########################TASK 2##############################3

#Get all the unique sessions
sessions = list(OrderedDict.fromkeys(X[:, 0]))
#print(len(sessions))

#Calculate the length of each session and the last index number of each session
i = 0
j, tot = 0, 0
total_session = [] 
session_end = []
for session in sessions:
    while i < len(X[:, 0]) and X[i, 0] == session:
        i += 1
    total_session.append(i - tot)
    tot = i
    session_end.append(i)
    j += 1
#print(session_end)   
#print(total_session)

#Create Graphs for each session
i = 0
j, k = 0, 0
Gs = []
G = nx.DiGraph()
for session in sessions:
    #load owners, commentators, comments, roles of each session
    if i == 0:
        owners = X[0:session_end[i], 3]
        commentors = X[0:session_end[i], 2]
        comments = X[0:session_end[i], 4]
        roles = X[0:session_end[i], 6]
    else:
        owners = X[session_end[i - 1]:session_end[i], 3]
        commentors = X[session_end[i - 1]:session_end[i], 2]
        comments = X[session_end[i - 1]:session_end[i], 4]
        roles = X[session_end[i - 1]:session_end[i], 6]
    #Names that start with @
    #Create an edge when node1 has addressed or directly replied to node2 
    for node1, node2 in zip(owners, commentors):
        G = addEdge(G, node2, node1, roles[j])
        node3 = asterisk_commentor(node2)
        for comment in comments:
            if node3 in comment:
                G = addEdge(G, commentors[k], removeAsterisk(node3), roles[k])
            k += 1
        k = 0
        j += 1
    j = 0
    Gs.append(G)
    G = nx.DiGraph()
    i += 1
#print(len(Gs))

#a) number of nodes, edges, and bidirectional edges, min, max, and average in–, out–and total degree

#Contains the number of nodes and edges of each graph Gs
num_of_nodes = []
num_of_edges = []

bidirectional_edges = []
min_values = []
max_values = []
average_degrees = []
in_degrees = []
out_degrees = []
total_degrees = []

#Contains all the degree values of every graph
all_degrees = {}
all_in_degrees = {}
all_out_degrees = {}
count = 0
for graph in Gs:
    num_of_nodes.append(len(graph.nodes()))
    num_of_edges.append(len(graph.edges()))
    #Calculate the total number of bidirectional edges
    for u in graph.nodes():
        for v in graph.nodes():
            if graph.has_edge(u, v) and graph.has_edge(v, u):
                count += 1
    bidirectional_edges.append(count)
    count = 0
    degree = dict(graph.degree())
    all_degrees = {**all_degrees, **degree}
    
    min_values.append(min(degree.values()))
    max_values.append(max(degree.values()))
    
    indegree = dict(graph.in_degree())
    all_in_degrees = {**all_in_degrees, **indegree}
    
    outdegree = dict(graph.out_degree())
    all_out_degrees = {**all_out_degrees, **outdegree}
    
    #Calculate and store the total, average, in and out degree of eacb Graph
    total_degrees.append(sum(degree.values()))
    average_degrees.append(sum(degree.values()) / float(len(graph.nodes())))
    in_degrees.append(sum(indegree.values()) / float(len(graph.nodes())))
    out_degrees.append(sum(outdegree.values()) / float(len(graph.nodes())))    
#Display Results
print('Total number of nodes: ', sum(num_of_nodes))
printMeanAndSD(num_of_nodes)

print('Total number of edges: ', sum(num_of_edges))
printMeanAndSD(num_of_edges)

print('Total number of bidirections edges: ', sum(bidirectional_edges))
printMeanAndSD(bidirectional_edges)

print('Min Degree: ', min(min_values))
printMeanAndSD(min_values)

print('Max Degree: ', max(max_values))
printMeanAndSD(max_values)

print('Total Degree: ', sum(average_degrees))
printMeanAndSD(average_degrees)

print('In-Degree: ', sum(in_degrees) / len(in_degrees))
printMeanAndSD(in_degrees)

print('Out-Degree: ', sum(out_degrees) / len(out_degrees))
printMeanAndSD(out_degrees)

print(len(all_out_degrees))
#b
#Calculate and plot the in- and out-and total degree distributions
indeg_uniq = sorted(set(all_in_degrees.values()))
indeg_hist = [list(all_in_degrees.values()).count(x) for x in indeg_uniq]

outdeg_uniq = sorted(set(all_out_degrees.values()))
outdeg_hist = [list(all_out_degrees.values()).count(x) for x in outdeg_uniq]

uniq = sorted(set(all_degrees.values()))
hist = [list(all_degrees.values()).count(x) for x in uniq]

#draw the corresponding best fitting least–square regression line
y_pred_in, coeffs = lr(indeg_uniq, indeg_hist)
y_pred_out, coeffs1 = lr(outdeg_uniq, outdeg_hist)
y_pred_total, coeffs2 = lr(uniq, hist)
#print (coeffs.size)

#Plot the degrees with the best fitting regression curve
fig = plt.figure(figsize=(15,10))
plt.scatter(indeg_uniq, indeg_hist, s=10, color = 'green',label='In-Degree')
plt.plot(indeg_uniq, y_pred_in, color = 'green', label=coeffs)
plt.scatter(outdeg_uniq, outdeg_hist, s=10,color = 'red',label='Out-Degree')
plt.plot(outdeg_uniq, y_pred_out, color = 'red', label=coeffs1)
plt.scatter(uniq, hist,s=10,color = 'yellow', label= 'Degree')
plt.plot(uniq, y_pred_total, color = 'yellow', label=coeffs2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.legend()
plt.title('In-,Out- and Degree Distributions')
plt.show()

bridge_count = []
local_bridge_count = []
c1, c2 = 0, 0
#C between a bully and a victim in Gs is a bridge or a local bridge
for graph in Gs:
    graph1 = graph.to_undirected()
    #Get all the bridges
    bridges = list(nx.bridges(graph1))
    #Check if a bully and a victim in Gs is a bridge or a local bridge
    for bridge in bridges:
        if IsBridgeBetweenBullyAndVictim(graph1, bridge):
            c1 += 1
    bridge_count.append(c1)
    c1 = 0
    #Get all the local bridges
    local_bridges = list(nx.local_bridges(graph1))
    for local_bridge in local_bridges:
        if IsBridgeBetweenBullyAndVictim(graph1, local_bridge):
            c2 += 1
    local_bridge_count.append(c2)
    c2 = 0                                                                                                                                                              
#print(sum(bridge_count))                                                                         
#print(sum(local_bridge_count))

#e Compute the principal eigenvalue, λws, of the weighted and undirected adjacency matrix of each Gs
print('Number of nodes of each graph: \n', num_of_nodes)
print('Number of edges of each graph: \n', num_of_nodes)
print('Total degree of each graph: \n', total_degrees)

#Convert graphs to undirected to calculate principal eigen value
Hs = []
for graph in Gs:
    h = graph.to_undirected()
    Hs.append(h)
lambdaws = []
for each in Hs:
    w, v = eigenvalues(each)
    lambdaws.append(w.tolist())

#Flatten the list of lists of the principal eigen values
f_lambdaws = [val for sublist in lambdaws for val in sublist]
#print(len(f_lambdaws))

#Plot the distribution of these values
uniq_nodes = sorted(set(num_of_nodes))
node_hist = [list(num_of_nodes).count(x) for x in uniq_nodes]
plotGraph(uniq_nodes, node_hist, '|Nodes|', 'Count', 'Node Distribution')

uniq_edges = sorted(set(num_of_edges))
edge_hist = [list(num_of_edges).count(x) for x in uniq_edges]
plotGraph(uniq_edges, edge_hist, '|Edges|', 'Count', 'Edge Distribution')

uniq_totaldeg = sorted(set(total_degrees))
totaldeg_hist = [list(total_degrees).count(x) for x in uniq_edges]
plotGraph(uniq_totaldeg, totaldeg_hist, 'Degree', 'Count', 'Total Degree Distribution')

#Plot the distribution of the eigenvalues
warnings.filterwarnings("ignore")

uniq_eigen = list(f_lambdaws)
totaleigen_hist = [list(f_lambdaws).count(x) for x in uniq_eigen]
plotGraph(uniq_eigen, totaleigen_hist, 'Eigen Values', 'Count', 'Principal eigenvalues Distribution')


#####################################  TASK 3 ################################
eegos = []
negos = []
feature_matrices = []
#|Eego(i)| (i.e., number of edges in node i’s egonet), and (ii) |N(ego(i))|: number of neighbors
# of ego(i) (i.e., node i’s egonet)
for graph in Gs:
    nodes1 = graph.nodes()
    for node in nodes1:
        edges1 = len(graph.edges(node))
        neighbors = len(list(graph.neighbors(node)))
        eegos.append(edges1)
        negos.append(neighbors)
    #Create a feature matrix for each graphs
    feature_matrix = np.array([nodes1, eegos, negos])
    feature_matrices.append(feature_matrix)
    eegos = []
    negos = []
#print(feature_matrices[0])   

#B Compute the median, mean, and standard deviation of each feature
SGs = []
for features in feature_matrices:
    fe1 = features[1].astype(int)
    fe2 = features[2].astype(int)
    #Calculate the mean, median and standard deviation for each feature 
    f1 = [mean(fe1), median(fe1), stdev(fe1)]
    f2 = [mean(fe2), median(fe2), stdev(fe2)]
    #Store the resulted feature vectors into one for each graph
    merge = f1 + f2
    SGs.append(merge)
print('The Single Feature Vectors for each graph are: \n',SGs)