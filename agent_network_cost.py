import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import expit 

# weight factors for centrality measures
alpha, beta, gamma, epsilon = 0.35, 0.25, 0.25, 0.15  
# priority error weight
p = 0.7  

## source data files. More layers can be added and formulas updated 
## i.e. security threats, resource depth and churn rates, 
# deterministic uptime/downtime metrics with SLA thresholds etc.

adjacency_csv = "adjacency_matrix.csv" 
error_thresholds_csv = "error_policy.csv"
cost_csv = "cost.csv"

# graph building
adj_matrix = pd.read_csv(adjacency_csv, index_col=0)
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# build error threshold data
error_df = pd.read_csv(error_thresholds_csv)

# error policy
type1_error = dict(zip(error_df["Node"], error_df["Type-1 Error"]))
type2_error = dict(zip(error_df["Node"], error_df["Type-2 Error"]))
priority = dict(zip(error_df["Node"], error_df["Priority"]))

# cost data
cost_df = pd.read_csv(cost_csv)

#cost parameters
success_value = dict(zip(cost_df["Node"], cost_df["Success_Value"]))
error_cost = dict(zip(cost_df["Node"], cost_df["Error_Cost"]))
revenue_unit = dict(zip(cost_df["Node"], cost_df["Revenue_Unit"]))

# centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
closeness_centrality = nx.closeness_centrality(G)

# normalize the centrality scores 
def normalize(data):
    min_val, max_val = min(data.values()), max(data.values())
    return {key: (val - min_val) / (max_val - min_val) if max_val > min_val else 0 for key, val in data.items()}

norm_degree_centrality = normalize(degree_centrality)
norm_betweenness_centrality = normalize(betweenness_centrality)
norm_eigenvector_centrality = normalize(eigenvector_centrality)
norm_closeness_centrality = normalize(closeness_centrality)

# Raw Risk Score (R) from Centrality 
raw_risk_scores = {
    node: (alpha * norm_degree_centrality[node] + 
           beta * norm_betweenness_centrality[node] + 
           gamma * norm_eigenvector_centrality[node] +
           epsilon * norm_closeness_centrality[node])
    for node in G.nodes()
}

# optional Sigmoid transformation to Risk Score
# if intuition is there might be something more nuanced to capture, nonlinear etc.  
transformed_risk_scores = {node: expit(raw_risk_scores[node]) for node in G.nodes()}

# weighted error rates
weighted_error_rate = {
    node: (p * type1_error[node] + (1 - p) * type2_error[node]) if priority[node] == 1 else
          (p * type2_error[node] + (1 - p) * type1_error[node])
    for node in G.nodes()
}

# FinalRiskScore
final_risk_scores = {
    node: transformed_risk_scores[node] * weighted_error_rate[node]
    for node in G.nodes()
}

# cost impact
cost_impact = {
    node: final_risk_scores[node] * error_cost[node] + (1 - final_risk_scores[node]) * success_value[node]
    for node in G.nodes()
}

# revenue impact
total_success_revenue = {
    node: (1 - final_risk_scores[node]) * success_value[node] * revenue_unit[node]
    for node in G.nodes()
}

total_error_loss = {
    node: final_risk_scores[node] * error_cost[node] * revenue_unit[node]
    for node in G.nodes()
}

net_revenue_impact = {
    node: total_success_revenue[node] + total_error_loss[node]
    for node in G.nodes()
}

# prepare data for visualize
impact_df = pd.DataFrame({
    "Final Risk Score": final_risk_scores,
    "Cost Impact": cost_impact,
    "Total Success Revenue": total_success_revenue,
    "Total Error Loss": total_error_loss,
    "Net Revenue Impact": net_revenue_impact
}).round(4)

# results
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("tight")
ax.axis("off")
ax.table(cellText=impact_df.values, colLabels=impact_df.columns, rowLabels=impact_df.index, cellLoc="center", loc="center")

# plottng
plt.title("AI Agent Risk, Cost, and Revenue Impact Analysis", fontsize=12, fontweight="bold")
plt.show()
