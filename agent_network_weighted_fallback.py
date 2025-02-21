# Re-import necessary libraries since execution state was reset
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load adjacency matrix from CSV
csv_file = "test_adj_matrix.csv"  # Adjust with actual filename
adj_matrix = pd.read_csv(csv_file, index_col=0) 

# Convert DataFrame to NetworkX directed graph
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# Load node weights from CSV
weights_df = pd.read_csv("node_fallback_weights.csv")  # Read node weights file
node_weights = dict(zip(weights_df["Node"], weights_df["Handling_Probability"]))


# Normalize weights for visualization (higher weight = darker node color)
norm = plt.Normalize(min(node_weights.values()), max(node_weights.values()))
node_colors_gradient = [plt.cm.Blues(norm(node_weights[node])) for node in G.nodes()]

# Draw the weighted graph with node coloring based on handling probability
plt.figure(figsize=(6, 4))
pos_viz = nx.spring_layout(G, seed=42)  # Layout for visualization

nx.draw(G, pos_viz, with_labels=True, 
        node_color=node_colors_gradient, 
        edge_color='black', node_size=2000, font_size=12, arrows=True)

# Add color bar for node weights
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.2, pad=0.08)
cbar.set_label("Node Direct Handling Probability")

plt.subplots_adjust(right=0.85)
# Display the graph
plt.title("Node Network with Direct Handling Probabilities")
plt.show()
