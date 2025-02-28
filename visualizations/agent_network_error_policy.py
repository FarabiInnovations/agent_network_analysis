import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#  adjacency matrix
csv_file = "../adjacency_matrices/adjacency_matrix.csv" # can use directed, undirected, weighted, mixed graph 
adj_matrix = pd.read_csv(csv_file, index_col=0) 

# grapsh build
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# error policy
error_df = pd.read_csv("../error_policies/error_policy.csv")

# grab relevant columns
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

# node coloring logic based on priority and error thresholds
node_colors = []
for node in G.nodes():
    if compliance[node]:
        node_colors.append("green")  # Compliant nodes remain green
    else:
        if (priority[node] == 1 and type1_error[node] > max_type1[node]) or \
           (priority[node] == 2 and type2_error[node] > max_type2[node]):
            node_colors.append("red")  # Threshold exceeded for prioritized error
        else:
            node_colors.append("orange")  # Otherwise, flag (warning level)

# error labels to display only breached thresholds given priority
error_labels = {}
for node in G.nodes():
    if not compliance[node]:  # apply only labels for non-compliant nodes
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


plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G, seed=42)


nx.draw(G, pos, with_labels=True, 
        node_color=node_colors, edge_color='black', 
        node_size=2000, font_size=12, arrows=True)  

# Add error labels **only for non-compliant nodes**, positioned slightly to the right
for node, label in error_labels.items():
    plt.text(pos[node][0] - 0.03, pos[node][1] - 0.04, label, fontsize=8, color="black",
             verticalalignment='center', horizontalalignment='right', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle="round,pad=0.3"))


plt.title("AI Agent Network Threshold-Based Error Thresholds \n(Green = Compliant, Red = Severe, Orange = Warning)")
plt.show()