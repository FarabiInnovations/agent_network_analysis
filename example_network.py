import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load adjacency matrix from CSV
csv_file = "example_adjancency_matrix.csv"  # Adjust with actual filename
adj_matrix = pd.read_csv(csv_file, index_col=0)  # Set first column as row index

# Convert DataFrame to NetworkX directed graph
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# Define node colors: Green for user-facing nodes (self-loop present), Blue for others
node_colors_viz = {node: "green" if adj_matrix.loc[node, node] == 1 else "blue" for node in G.nodes()}

# Draw the graph
plt.figure(figsize=(6, 4))
pos_viz = nx.spring_layout(G, seed=42)  # Layout for visualization

nx.draw(G, pos_viz, with_labels=True, 
        node_color=[node_colors_viz[node] for node in G.nodes()], 
        edge_color='black', node_size=2000, font_size=12, font_color='yellow', arrows=True)

# Display the graph
plt.title("Visualization of Sample Node Network")
plt.show()

# Visualize adjacency matrix
#print(adj_matrix)

# Print graph edges
#print("Graph edges:", list(G.edges()))
