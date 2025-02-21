import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# Load adjacency matrix from CSV
csv_file = "adjacency_matrix.csv"  # Adjust with actual filename
adj_matrix = pd.read_csv(csv_file, index_col=0)  # Set first column as row index

# Convert adjacency matrix to NetworkX graph
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# Load error threshold data from CSV
error_df = pd.read_csv("error_policy.csv")

# Extract relevant columns
accuracy = dict(zip(error_df["Node"], error_df["Accuracy"]))
type1_error = dict(zip(error_df["Node"], error_df["Type-1 Error"]))
type2_error = dict(zip(error_df["Node"], error_df["Type-2 Error"]))
min_accuracy = dict(zip(error_df["Node"], error_df["Min_Accuracy"]))
max_type1 = dict(zip(error_df["Node"], error_df["Max_Type-1"]))
max_type2 = dict(zip(error_df["Node"], error_df["Max_Type-2"]))
priority = dict(zip(error_df["Node"], error_df["Priority"]))  # Priority flag

# Compute compliance (whether the node meets business error policy)
compliance = {
    node: (accuracy[node] >= min_accuracy[node]) and 
          (type1_error[node] <= max_type1[node]) and 
          (type2_error[node] <= max_type2[node])
    for node in G.nodes()
}

# Adjust node coloring logic based on priority and error thresholds
node_colors = []
for node in G.nodes():
    if compliance[node]:
        node_colors.append("green")  # Compliant nodes remain green
    else:
        if (priority[node] == 1 and type1_error[node] > max_type1[node]) or \
           (priority[node] == 2 and type2_error[node] > max_type2[node]):
            node_colors.append("red")  # Threshold exceeded for prioritized error → Red
        else:
            node_colors.append("orange")  # Otherwise, flag as Orange (warning level)

# Adjust error labels to display only the breached threshold based on priority
error_labels = {}
for node in G.nodes():
    if not compliance[node]:  # Only apply labels for non-compliant nodes
        if priority[node] == 1 and type1_error[node] > max_type1[node]:  # Type-1 is more severe
            error_labels[node] = f"T1: {type1_error[node]:.2f} (>{max_type1[node]:.2f})"
        elif priority[node] == 2 and type2_error[node] > max_type2[node]:  # Type-2 is more severe
            error_labels[node] = f"T2: {type2_error[node]:.2f} (>{max_type2[node]:.2f})"
        else:
            # If both errors are above the threshold (unlikely but possible), show both
            breached_errors = []
            if type1_error[node] > max_type1[node]:
                breached_errors.append(f"T1: {type1_error[node]:.2f} (>{max_type1[node]:.2f})")
            if type2_error[node] > max_type2[node]:
                breached_errors.append(f"T2: {type2_error[node]:.2f} (>{max_type2[node]:.2f})")
            error_labels[node] = "\n".join(breached_errors)

# Draw the network with updated error threshold coloring
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G, seed=42)

# Draw nodes with updated colors
nx.draw(G, pos, with_labels=True, 
        node_color=node_colors, edge_color='black', 
        node_size=2000, font_size=12, arrows=True)  # Larger font for main node labels

# Add error labels **only for non-compliant nodes**, positioned slightly to the right
for node, label in error_labels.items():
    plt.text(pos[node][0] - 0.03, pos[node][1] - 0.04, label, fontsize=8, color="black",
             verticalalignment='center', horizontalalignment='right', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle="round,pad=0.3"))

# Add title
plt.title("AI Agent Network Threshold-Based Error Thresholds \n(Green = Compliant, Red = Severe, Orange = Warning)")
plt.show()