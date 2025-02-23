import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import expit 

# Weight factors for centrality measures
alpha, beta, gamma, epsilon = 0.35, 0.25, 0.25, 0.15  
p = 0.7  # priority error weight

# Data files
adjacency_csv = "adjacency_matrix.csv" 
error_thresholds_csv = "error_policy.csv"
cost_csv = "cost.csv"

# additional risk factors
resource_threats_csv = "resource_threats.csv"  # columns: resource_threat_ID, resources_available, churn_rate
pattern_threats_csv = "design_threats.csv"     # columns: Design_ID, Design, Threat, Level
SLA_csv = "SLA.csv"                             
node_threat_map_csv = "node_threat_map.csv"    # columns: node, design_ID, resource_threat_ID


# Build graph
adj_matrix = pd.read_csv(adjacency_csv, index_col=0)
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# Build error threshold data
error_df = pd.read_csv(error_thresholds_csv)
type1_error = dict(zip(error_df["Node"], error_df["Type-1 Error"]))
type2_error = dict(zip(error_df["Node"], error_df["Type-2 Error"]))
priority = dict(zip(error_df["Node"], error_df["Priority"]))

# Cost data
cost_df = pd.read_csv(cost_csv)
success_value = dict(zip(cost_df["Node"], cost_df["Success_Value"]))
error_cost = dict(zip(cost_df["Node"], cost_df["Error_Cost"]))
revenue_unit = dict(zip(cost_df["Node"], cost_df["Revenue_Unit"]))

# Centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
closeness_centrality = nx.closeness_centrality(G)

# Normalize the centrality scores 
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

# Optional Sigmoid transformation to Risk Score
transformed_risk_scores = {node: expit(raw_risk_scores[node]) for node in G.nodes()}

# Weighted error rates based on error policy
weighted_error_rate = {
    node: (p * type1_error[node] + (1 - p) * type2_error[node]) if priority[node] == 1 else
          (p * type2_error[node] + (1 - p) * type1_error[node])
    for node in G.nodes()
}

# Baseline final risk score from centrality and error policies
final_risk_scores = {
    node: transformed_risk_scores[node] * weighted_error_rate[node]
    for node in G.nodes()
}

# --------------------------
# Incorporate New Risk Factors
# --------------------------

# Load new risk factor data
resource_threats_df = pd.read_csv(resource_threats_csv)
pattern_threats_df = pd.read_csv(pattern_threats_csv)
node_threat_map_df = pd.read_csv(node_threat_map_csv)
uptime_df = pd.read_csv(SLA_csv)

# Create dictionaries for actual uptime and SLA uptime (make sure these are expressed as decimals, e.g., 0.99 for 99%)
# Convert SLA threats into a dictionary
SLA_threats = {
    row['SLA_ID']: {
        'sla_uptime': row['sla_uptime'], 
        'actual_uptime': row['actual_uptime']
    }
    for _, row in uptime_df.iterrows()
}

# Convert resource threats into a dictionary
# Example row: resource_threat_ID, resources_available, churn_rate
resource_threats = {
    row['resource_threat_ID']: {
        'resources_available': row['resources_available'], 
        'churn_rate': row['churn_rate']
    }
    for _, row in resource_threats_df.iterrows()
}

# Convert pattern threats into a dictionary (using Level as the threat magnitude)
# Example row: Design_ID, Design, Threat, Level
pattern_threats = {row['Design_ID']: row['Level'] for _, row in pattern_threats_df.iterrows()}

# Convert node threat mapping: maps a node to its design and resource threat IDs
# Example row: node, design_ID, resource_threat_ID
node_threat_map = {
    row['node']: {'design_ID': row['design_ID'], 'resource_threat_ID': row['resource_threat_ID'],
     'SLA_ID': row['SLA_ID']}
    for _, row in node_threat_map_df.iterrows()
}

# Option flag: if True, use additive approach for resource risk; otherwise, use multiplicative.
use_resource_additive = False

# Adjust final risk score to incorporate design pattern and resource risks.
# For design threats: we use a multiplicative factor (e.g., multiplier = 1 + Level)
# For resource risk (churn): by default, use multiplicative approach with a factor of (churn_rate^resources_available)
#   (i.e., the probability that all resources fail); or optionally, add an extra risk term.
adjusted_final_risk = {}
for node in G.nodes():
    base = final_risk_scores[node]
    
    # Default multipliers
    pattern_multiplier = 1.0
    resource_multiplier = 1.0
    uptime_multiplier = 1.0
    resource_additive_term = 0.0
    
    if node in node_threat_map:
        mapping = node_threat_map[node]
        design_id = mapping['design_ID']
        resource_threat_id = mapping['resource_threat_ID']
        sla_id = mapping['SLA_ID']
        
        # Pattern threat multiplier (multiplicative approach)
        if design_id in pattern_threats:
            level = pattern_threats[design_id]
            # For example, a Level of 0.5 increases risk by 50%
            pattern_multiplier = 1 + level
        
        # Resource risk: compute risk reduction from resources.
        if resource_threat_id in resource_threats:
            res_data = resource_threats[resource_threat_id]
            resources_available = res_data['resources_available']
            churn_rate = res_data['churn_rate']
            
            # Multiplicative approach:
            # Here, using churn_rate^resources_available gives the probability that ALL resources fail.
            # A lower value indicates better coverage and thus lower risk.
            resource_multiplier = churn_rate ** resources_available
            
            # Optional additive approach: define an additive term (for instance, using the survival probability)
            # Survival probability is 1 - (churn_rate^resources_available)
            if use_resource_additive:
                resource_additive_term = 1 - (churn_rate ** resources_available)

        if sla_id in SLA_threats:
            res_data = SLA_threats[sla_id]
            sla_uptime = res_data['sla_uptime']
            actual_uptime = res_data['actual_uptime']
            
            if actual_uptime < sla_uptime:
              uptime_risk = (sla_uptime - actual_uptime) / sla_uptime
              uptime_multiplier = 1 + (uptime_risk)
                
    if use_resource_additive:
        # In the additive model, we add the resource risk term to the product of the baseline and pattern multiplier.
        adjusted_final_risk[node] = (base * pattern_multiplier * uptime_multiplier) + resource_additive_term
    else:
        # Multiplicative approach: multiply baseline risk by both multipliers.
        adjusted_final_risk[node] = base * pattern_multiplier * resource_multiplier * uptime_multiplier

# Now, 'adjusted_final_risk' is the risk probability for each node after incorporating centrality, error policies,
# design pattern threats, and resource risk factors.
#
# --------------------------
# Contagion Monte Carlo Simulation using adjusted_final_risk
# --------------------------

# Define contagion parameters
num_rounds = 2            # rounds to propagate contagion effects per simulation run
contagion_factor = 0.1    # how much to add to a node's baseline risk per fraction of failed neighbors

# Initialize simulation storage
num_simulations = 10000  # simulation runs
nodes = list(adjusted_final_risk.keys())
simulation_results = {node: [] for node in nodes}
network_net_revenue = []  # store net revenue per simulation across all nodes

for sim in range(num_simulations):
    # For each simulation run, simulate contagion rounds.
    node_failed = {node: False for node in nodes}
    # Set current risk per node from the adjusted risk
    current_risk = {node: adjusted_final_risk[node] for node in nodes}
    
    # Propagate failures over rounds
    for r in range(num_rounds):
        # Determine failure for nodes that haven't failed
        for node in nodes:
            if not node_failed[node]:
                if np.random.rand() < current_risk[node]:
                    node_failed[node] = True
        
        # Update risk for nodes that haven't failed based on neighbor failures
        for node in nodes:
            if not node_failed[node]:
                neighbors = list(G.neighbors(node))
                if neighbors:
                    fraction_failed = np.mean([node_failed.get(neighbor, False) for neighbor in neighbors])
                    updated_risk = adjusted_final_risk[node] + contagion_factor * fraction_failed
                    current_risk[node] = min(updated_risk, 1.0)  # cap at 1
                else:
                    current_risk[node] = adjusted_final_risk[node]
    
    # Calculate net revenue outcome per node for this simulation run
    sim_net_revenue = 0
    for node in nodes:
        if node_failed[node]:
            outcome = error_cost[node] * revenue_unit[node]
        else:
            outcome = success_value[node] * revenue_unit[node]
        simulation_results[node].append(outcome)
        sim_net_revenue += outcome
    network_net_revenue.append(sim_net_revenue)

# Convert simulation results to DataFrame
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
