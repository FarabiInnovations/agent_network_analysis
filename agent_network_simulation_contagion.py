import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import expit 

#weight factors for centrality measures
alpha, beta, gamma, epsilon = 0.35, 0.25, 0.25, 0.15  
p = 0.7  # priority error weight

# data
adjacency_csv = "adjacency_matrix.csv" 
error_thresholds_csv = "error_policy.csv"
cost_csv = "cost.csv"

# build graph
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

# FinalRiskScore
final_risk_scores = {
    node: transformed_risk_scores[node] * weighted_error_rate[node]
    for node in G.nodes()
}

# Define contagion parameters
num_rounds = 2            # number of rounds to propagate contagion effects per simulation run
contagion_factor = 0.1   # how much to add to a node baseline risk per fraction of failed neighbors

# 'final_risk_scores' as the baseline risk probabilities.
# Initialize simulation storage
num_simulations = 10000  # simulation runs
nodes = list(final_risk_scores.keys())
simulation_results = {node: [] for node in nodes}
network_net_revenue = []  # store net revenue per simulation across all nodes

for sim in range(num_simulations):
    # For each simulation run, simulate in rounds to capture contagion
    # track which nodes have failed
    node_failed = {node: False for node in nodes}
    
    # baseline risk for each node.
    current_risk = {node: final_risk_scores[node] for node in nodes}
    
    # propagate failures in rounds.
    for r in range(num_rounds):
        # for each node that hasn't failed, determine if an error occurs using the current risk.
        for node in nodes:
            if not node_failed[node]:
                if np.random.rand() < current_risk[node]:
                    node_failed[node] = True

        # after round, update the risk probabilities for nodes that haven't failed,
        # based on failure status of neighbors.
        for node in nodes:
            if not node_failed[node]:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    #fraction of neighbors that failed.
                    fraction_failed = np.mean([node_failed.get(neighbor, False) for neighbor in neighbors])
                    # Increase the risk: baseline risk plus contagion increment.
                    updated_risk = final_risk_scores[node] + contagion_factor * fraction_failed
                    current_risk[node] = min(updated_risk, 1.0)  # capped at 1
                else:
                    # If no neighbors, risk remains baseline.
                    current_risk[node] = final_risk_scores[node]

    # Once rounds are completed each node is either failed (error) or not.
    sim_net_revenue = 0  # init network net revenue for run
    for node in nodes:
        if node_failed[node]:
            outcome = error_cost[node] * revenue_unit[node]
        else:
            outcome = success_value[node] * revenue_unit[node]
        simulation_results[node].append(outcome)
        sim_net_revenue += outcome
    network_net_revenue.append(sim_net_revenue)

# results to DataFrame 
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
plt.title("Distribution of Network Net Revenue Impact (Monte Carlo Simulation with Contagion)")
plt.show()

