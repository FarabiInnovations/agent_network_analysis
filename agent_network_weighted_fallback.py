import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# adjacency matrix
csv_file = "test_adj_matrix.csv"  
adj_matrix = pd.read_csv(csv_file, index_col=0) 

# build graph
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# node weights 
weights_df = pd.read_csv("node_fallback_weights.csv") 
node_weights = dict(zip(weights_df["Node"], weights_df["Handling_Probability"]))



norm = plt.Normalize(min(node_weights.values()), max(node_weights.values()))
node_colors_gradient = [plt.cm.Blues(norm(node_weights[node])) for node in G.nodes()]


plt.figure(figsize=(6, 4))
pos_viz = nx.spring_layout(G, seed=42) 

nx.draw(G, pos_viz, with_labels=True, 
        node_color=node_colors_gradient, 
        edge_color='black', node_size=2000, font_size=12, arrows=True)


sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.2, pad=0.08)
cbar.set_label("Node Direct Handling Probability")

plt.subplots_adjust(right=0.85)
plt.title("Node Network with Direct Handling Probabilities")
plt.show()
