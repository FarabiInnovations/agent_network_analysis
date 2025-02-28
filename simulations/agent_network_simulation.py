##### this test assumes node idenpendance ######

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import expit  

#weight factors for centrality measures
alpha, beta, gamma, epsilon = 0.35, 0.25, 0.25, 0.15  
p = 0.7  # priority error weight

# Data
adjacency_csv = "../adjacency_matrices/adjacency_matrix.csv" 
error_thresholds_csv = "../error_policies/error_policy.csv"
cost_csv = "../economic_impact/cost.csv"

# Build graph
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
# if intuition is there might be something more nuanced to capture 
transformed_risk_scores = {node: expit(raw_risk_scores[node]) for node in G.nodes()}

# weighted error rates
weighted_error_rate = {
    node: (p * type1_error[node] + (1 - p) * type2_error[node]) if priority[node] == 1 else
          (p * type2_error[node] + (1 - p) * type1_error[node])
    for node in G.nodes()
}

# FinalRiskScore. We can add more risk factors if needed e.g.
# security_risk = { node: betweenness_centrality[node] * injection_surface[node] for node in G.nodes() }
# final_risk_scores[node] += security_risk[node] * threat_intelligence[node]
##
final_risk_scores = {
    node: transformed_risk_scores[node] * weighted_error_rate[node]
    for node in G.nodes()
}

num_simulations = 10000  # simulation runs
nodes = list(final_risk_scores.keys())

# dictionary to store aggregated simulation results per node and overall network
simulation_results = {node: [] for node in nodes}
network_net_revenue = []  # store net revenue per simulation across all nodes

for sim in range(num_simulations):
    sim_net_revenue = 0  # aggregate net revenue for this run
    for node in nodes:
        # determine per node if an error occurs based on its risk probability.
        risk_prob = final_risk_scores[node]
        error_occurred = np.random.rand() < risk_prob 
        
        # calculate revenue outcome:
        # if error occurs, we get an error_cost (which is negative)
        # if no error, we get success_value.
        # then multiply by revenue_unit to scale it appropriately.
        # This can be any unit of value 'revenue' is the toy example, this could be
        # time etc. any unit of value determined to be relevant to measure 
        if error_occurred:
            outcome = error_cost[node] * revenue_unit[node]
        else:
            outcome = success_value[node] * revenue_unit[node]
        
        simulation_results[node].append(outcome)
        sim_net_revenue += outcome
    network_net_revenue.append(sim_net_revenue)


sim_df = pd.DataFrame(simulation_results)
sim_df['Network_Net_Revenue'] = network_net_revenue


network_mean = np.mean(network_net_revenue)
network_std = np.std(network_net_revenue)
print(f"Network Mean Net Revenue: {network_mean:.4f}")
print(f"Network Revenue Standard Deviation: {network_std:.4f}")


plt.figure(figsize=(8, 4))
plt.hist(network_net_revenue, bins=50, alpha=0.7, color='skyblue')
plt.xlabel("Net Revenue Impact")
plt.ylabel("Frequency")
plt.title("Distribution of Network Net Revenue Impact (MC Sim)")
plt.show()