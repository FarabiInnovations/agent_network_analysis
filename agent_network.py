# Define node colors: Green for user-facing agent (A), Blue for others
node_colors = {node: "green" if node == "A" else "blue" for node in nodes}

# Create a directed graph with updated edges including self-loop for A
G_user_facing = nx.DiGraph()
for source, targets in edges_with_cycles_dict.items():
    for target in targets:
        G_user_facing.add_edge(source, target)
G_user_facing.add_edge("A", "A")  # Adding the self-loop for user-facing indication

# Draw the directed graph with color distinction
plt.figure(figsize=(5, 3))
pos = nx.spring_layout(G_user_facing, seed=42)  # Layout for better visualization
nx.draw(G_user_facing, pos, with_labels=True, 
        node_color=[node_colors[node] for node in G_user_facing.nodes], 
        edge_color='black', node_size=2000, font_size=12, arrows=True)

# Display the graph
plt.title("Directed Graph with User-Facing Agent (Green)")
plt.show()

## hard coded tests
# Define the updated adjacency matrix with B and A as user-facing (green), and B, C feeding back to A
nodes_new = ["A", "B", "C", "D", "E"]
adj_matrix_new = pd.DataFrame(0, index=nodes_new, columns=nodes_new)

# Updated directed edges based on the new adjacency matrix
edges_new = [
    ("A", "A"),  # A is user-facing
    ("A", "B"),
    ("B", "A"),  # B can feed back to A
    ("B", "B"),  # B is also user-facing
    ("B", "C"),
    ("C", "A"),  # C can feed back to A
    ("C", "D"),
    ("D", "E"),
    ("E", "C")   # E to C
]

# Create a directed graph with the updated edges
G_new = nx.DiGraph()
G_new.add_edges_from(edges_new)

# Define node colors: Green for user-facing agents (A and B), Blue for others
node_colors_new = {node: "green" if node in ["A", "B"] else "blue" for node in nodes_new}

# Draw the updated directed graph
plt.figure(figsize=(5, 3))
pos_new = nx.spring_layout(G_new, seed=42)  # Layout for visualization
nx.draw(G_new, pos_new, with_labels=True, 
        node_color=[node_colors_new[node] for node in G_new.nodes], 
        edge_color='black', node_size=2000, font_size=12, arrows=True)

# Display the graph
plt.title("Directed Graph with A and B as User-Facing (Green), Feedback from B and C to A")
plt.show()


##
import numpy as np

# Normalize error rates for color mapping (higher error = darker color)
error_values = np.array([error_policies[node] for node in nodes_new])
norm = plt.Normalize(error_values.min(), error_values.max())
node_colors_gradient = [plt.cm.Reds(norm(error_policies[node])) for node in nodes_new]

# Extract edge weights for visualization
edge_labels = {(edge[0], edge[1]): f"{weight:.2f}" for edge, weight in edge_weights.items()}

# Draw the weighted graph with error-based gradient coloring for nodes
plt.figure(figsize=(6, 4))
pos_weighted = nx.spring_layout(G_new, seed=42)  # Layout for visualization

nx.draw(G_new, pos_weighted, with_labels=True, 
        node_color=node_colors_gradient, 
        edge_color='black', node_size=2000, font_size=12, arrows=True, width=1.5)

# Draw edge labels for weights
nx.draw_networkx_edge_labels(G_new, pos_weighted, edge_labels=edge_labels, font_size=10)

# Add color bar for error rate mapping
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.04)
cbar.set_label("Node Error Rate (%)")

# Display the graph
plt.title("Weighted Directed Graph with Error-Based Node Coloring")
plt.show()

