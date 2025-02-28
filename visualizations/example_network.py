import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


csv_file = "../adjacency_matrices/example_adjacency_matrix.csv"  
adj_matrix = pd.read_csv(csv_file, index_col=0)  
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

node_colors_viz = {node: "green" if adj_matrix.loc[node, node] == 1 else "blue" for node in G.nodes()}


plt.figure(figsize=(6, 4))
pos_viz = nx.spring_layout(G, seed=42)  

nx.draw(G, pos_viz, with_labels=True, 
        node_color=[node_colors_viz[node] for node in G.nodes()], 
        edge_color='black', node_size=2000, font_size=12, font_color='yellow', arrows=True)


plt.title("Visualization of Sample Node Network")
plt.show()

##
#print(adj_matrix)
##
#print("Graph edges:", list(G.edges()))
