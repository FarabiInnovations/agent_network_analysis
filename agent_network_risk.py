import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Weight factors for centrality measures (adjusted to sum to 1 in this case
# Use intituition, evidence, or priors derived from literature to set weighting

alpha, beta, gamma, epsilon = 0.35, 0.25, 0.25, 0.15  
# Weights for priority error dominance. This wil be applied to the priority error
# and 1 - p will be applied to the non priority error
p = 0.7

# adjacency matrix
csv_file = "adjacency_matrix.csv"  
adj_matrix = pd.read_csv(csv_file, index_col=0) 

# cost 
cost_df = pd.read_csv("cost.csv")
success_value = dict(zip(cost_df["Node"], cost_df["Success_Value"]))
error_cost = dict(zip(cost_df["Node"], cost_df["Error_Cost"]))
revenue_unit = dict(zip(cost_df["Node"], cost_df["Revenue_Unit"]))

# build graph
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# error policy
error_df = pd.read_csv("error_policy.csv")
accuracy = dict(zip(error_df["Node"], error_df["Accuracy"]))
type1_error = dict(zip(error_df["Node"], error_df["Type-1 Error"]))
type2_error = dict(zip(error_df["Node"], error_df["Type-2 Error"]))
min_accuracy = dict(zip(error_df["Node"], error_df["Min_Accuracy"]))
max_type1 = dict(zip(error_df["Node"], error_df["Max_Type-1"]))
max_type2 = dict(zip(error_df["Node"], error_df["Max_Type-2"]))
priority = dict(zip(error_df["Node"], error_df["Priority"]))  # Priority flag

# centrality metrics
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Raw Risk Score (R) using the centrality metrics
raw_risk_scores = {
    node: (alpha * degree_centrality[node] + 
           beta * betweenness_centrality[node] + 
           gamma * eigenvector_centrality[node] +
           epsilon * closeness_centrality[node])
    for node in G.nodes()
}

# Final Risk Score incorporating error severity and priority weighting 
final_risk_scores = {
    node: raw_risk_scores[node] * (
        p * (type1_error[node] if priority[node] == 1 else type2_error[node]) + 
        (1 - p) * (type2_error[node] if priority[node] == 1 else type1_error[node])
    )
    for node in G.nodes()
}

# dataFrame for visualization
risk_df = pd.DataFrame({
    "Degree Centrality": degree_centrality,
    "Betweenness Centrality": betweenness_centrality,
    "Eigenvector Centrality": eigenvector_centrality,
    "Closeness Centrality": closeness_centrality,
    "Raw Risk Score": raw_risk_scores,
    "Final Risk Score": final_risk_scores,
}).round(4)


fig, ax = plt.subplots(figsize=(7, 4))


ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False) 
ax.set_frame_on(False)

table = ax.table(cellText=risk_df.values, 
                 colLabels=risk_df.columns,
                 rowLabels=risk_df.index, 
                 cellLoc="center", loc="center")

# table styls
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.2)  

#plot
plt.title("AI Agent Risk Scores", fontsize=12, fontweight="bold")
# -- #
plt.show()


