import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# adjacency matrix 
csv_file = "../adjacency_matrices/test_adj_matrix.csv"  
adj_matrix = pd.read_csv(csv_file, index_col=0) 

# build graph
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# Define node colors: 
# Toy example uses green for user-facing nodes (self-loop present), blue for others
# To use a self-loop like this in practice is up to you, it makes for a concise visual
# but user input nodes might want their own representation if the objective is to model
# the application piece separate from the LLM output with its own SLA policy with specific
# deterministic uptime metrics 

node_colors_viz = {node: "green" if adj_matrix.loc[node, node] == 1 else "blue" for node in G.nodes()}

# graph
plt.figure(figsize=(6, 4))
pos_viz = nx.spring_layout(G, seed=42)  

nx.draw(G, pos_viz, with_labels=True, 
        node_color=[node_colors_viz[node] for node in G.nodes()], 
        edge_color='black', node_size=2000, font_size=12, font_color='yellow', arrows=True)

#
plt.title("Visualization of Sample Node Network")
plt.show()

#adjacency matrix
#print(adj_matrix)

#graph edges
#print("Graph edges:", list(G.edges()))
