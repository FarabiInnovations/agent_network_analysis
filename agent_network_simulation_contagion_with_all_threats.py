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
resource_threats_csv = "resource_threats.csv"  
pattern_threats_csv = "design_threats.csv"    
SLA_csv = "SLA.csv"                             
node_threat_map_csv = "node_threat_map.csv" 


# graph
adj_matrix = pd.read_csv(adjacency_csv, index_col=0)
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# error threshold data
error_df = pd.read_csv(error_thresholds_csv)
type1_error = dict(zip(error_df["Node"], error_df["Type-1 Error"]))
type2_error = dict(zip(error_df["Node"], error_df["Type-2 Error"]))
priority = dict(zip(error_df["Node"], error_df["Priority"]))
null_flag = dict(zip(error_df["Node"], error_df["Ignore"]))

# cost data
cost_df = pd.read_csv(cost_csv)
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

# Optional Sigmoid transformation to Risk Score
transformed_risk_scores = {node: expit(raw_risk_scores[node]) for node in G.nodes()}

# Weighted error rates based on error policy
# If the error policy ignore flag = 1, that indicates that this is not a node with a
# distribution of outcomes (transformer output), and will ignore any type-1 
# type-2 error risk. All other risk metrics will apply. 
weighted_error_rate = {
    node: 0.0001 if null_flag[node] == 1 else (
        (p * type1_error[node] + (1 - p) * type2_error[node])
        if priority[node] == 1 else
        (p * type2_error[node] + (1 - p) * type1_error[node])
    )
    for node in G.nodes()
}

# Baseline final risk score from centrality and error policies
final_risk_scores = {
     node: transformed_risk_scores[node] if weighted_error_rate[node] == 99 
          else transformed_risk_scores[node] * weighted_error_rate[node]
    for node in G.nodes()
}

# --------------------------
# Additional Risk Factors
# --------------------------

# risk factor data
resource_threats_df = pd.read_csv(resource_threats_csv)
pattern_threats_df = pd.read_csv(pattern_threats_csv)
node_threat_map_df = pd.read_csv(node_threat_map_csv)
uptime_df = pd.read_csv(SLA_csv)

# dictionaries for the actual uptime and SLA uptime 
SLA_threats = {
    row['SLA_ID']: {
        'sla_uptime': row['sla_uptime'], 
        'actual_uptime': row['actual_uptime']
    }
    for _, row in uptime_df.iterrows()
}

# Resource threats dictionary
## -- ##
resource_threats = {
    row['resource_threat_ID']: {
        'resources_available': row['resources_available'], 
        'churn_rate': row['churn_rate']
    }
    for _, row in resource_threats_df.iterrows()
}

# Pattern threats dictionary (Level is the threat magnitude)
pattern_threats = {row['Design_ID']: row['Level'] for _, row in pattern_threats_df.iterrows()}

# Storing Agent Design Patterns for qualtitative analysis
pattern_details = {
    row['Design_ID'].strip(): {
        'Design': row['Design'].strip(),
        'Threat': row['Threat'].strip(),
        'Level': row['Level']
    }
    for _, row in pattern_threats_df.iterrows()
}

# This is the node threat mapping. This maps a node to its threat IDs
node_threat_map = {
    row['node']: {'design_ID': row['design_ID'], 'resource_threat_ID': row['resource_threat_ID'],
     'SLA_ID': row['SLA_ID']}
    for _, row in node_threat_map_df.iterrows()
}

# Option flag: if True, use additive approach for the resource risk, otherwise its multiplicative.
use_resource_additive = False

# Adjust final risk score.
# For agentic pattern threats it is a multiplicative factor (e.g., multiplier = 1 + Level)
# For resource risk (churn): by default, use multiplicative approach with a factor of
# (churn_rate^resources_available)
# SLA incorporates the SLA and actual uptime metrics
adjusted_final_risk = {}
risk_details = {}
for node in G.nodes():
    base = final_risk_scores[node]
    
    # default multipliers
    pattern_multiplier = 1.0
    resource_multiplier = 1.0
    uptime_multiplier = 1.0
    resource_additive_term = 0.0
    design_name = "n/a"
    threat_name = "n/a"

    
    if node in node_threat_map:
        mapping = node_threat_map[node]
        design_id = mapping['design_ID']
        resource_threat_id = mapping['resource_threat_ID']
        sla_id = mapping['SLA_ID']
        
        # Agentic Design Pattern threat multiplier (this could be any design threat
        # does not necessarily need to be Agentic Patterns)
        if design_id in pattern_threats:

            details = pattern_details[design_id]
            design_name = details['Design']
            threat_name = details['Threat']
            level = pattern_threats[design_id]
            
            pattern_multiplier = 1 + level
        
        # risk reduction from resources.
        if resource_threat_id in resource_threats:
            res_data = resource_threats[resource_threat_id]
            resources_available = res_data['resources_available']
            churn_rate = res_data['churn_rate']
            
            # Multiplicative approach:
            # using churn_rate^(resources_available/scaling factor) gives the probability that ALL resources fail
            # a lower value indicates better coverage and thus lower risk.
            scaling_factor = 10  # adjust as needed
            resource_multiplier = churn_rate ** (resources_available / scaling_factor)
            
            # Optional additive approach (this needs some adjusting)
            # Survival probability is 1 - (resources_available/scaling factor)
            if use_resource_additive:
                resource_multiplier = churn_rate ** (resources_available / scaling_factor)

        if sla_id in SLA_threats:
            res_data = SLA_threats[sla_id]
            sla_uptime = res_data['sla_uptime']
            actual_uptime = res_data['actual_uptime']
            
            if actual_uptime < sla_uptime:
              uptime_risk = (sla_uptime - actual_uptime) / sla_uptime
              uptime_multiplier = 1 + (uptime_risk)
                
    if use_resource_additive:
        # ** additive option, but not using and may remove this ** #
        adjusted_final_risk[node] = (base * pattern_multiplier * uptime_multiplier) + resource_additive_term
    else:
        # multiplicative approach
        print("****NODE START****")
        print("NODE KEY: ", node )
        print("Centrality Risk and Error Policy", base)
        print("Design Pattern Threats", pattern_multiplier)
        print("Resource and Churn Risk", resource_multiplier)
        print("SLA Uptime Risk", uptime_multiplier)
        print("***NODE END***")
        adjusted_final_risk[node] = base * pattern_multiplier * resource_multiplier * uptime_multiplier
        risk_details[node] = {
         'base_risk': round(base, 4),
         'pattern_multiplier': round(pattern_multiplier, 4),
         'resource_multiplier': round(resource_multiplier, 4),
         'uptime_multiplier': round(uptime_multiplier, 4),
         'adjusted_risk': round(adjusted_final_risk[node], 4),
         'Design_Name': design_name,
         'Threat_Name': threat_name
       }

# 'adjusted_final_risk' is the risk probability for each node after
# incorporating centrality, error policies,
# and all additional threats
#
# --------------------------
# Contagion Monte Carlo Simulation using adjusted_final_risk
# --------------------------

# contagion parameters
num_rounds = 2            # rounds to propagate contagion effects per simulation run
contagion_factor = 0.1    # how much to add to a node's baseline risk per fraction of failed neighbors

# Initialize simulation storage
num_simulations = 10000 
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
    
    #  net revenue outcome per node for this run
    sim_net_revenue = 0
    for node in nodes:
        if node_failed[node]:
            outcome = error_cost[node] * revenue_unit[node]
        else:
            outcome = success_value[node] * revenue_unit[node]
        simulation_results[node].append(outcome)
        sim_net_revenue += outcome
    network_net_revenue.append(sim_net_revenue)

# simulation results to DataFrame
sim_df = pd.DataFrame(simulation_results)
sim_df['Network_Net_Revenue'] = network_net_revenue

network_mean = np.mean(network_net_revenue)
network_std = np.std(network_net_revenue)

print("*****ENTERPRISE RESOURCES*****")
print(f"Network Mean Net Revenue: {network_mean:.4f}")
print(f"Network Revenue Standard Deviation: {network_std:.4f}")
print("******************************")

plt.figure(figsize=(8, 4))
plt.hist(network_net_revenue, bins=50, alpha=0.7, color='skyblue')
plt.xlabel("Net Revenue Impact")
plt.ylabel("Frequency")
plt.title("Distribution of Network Net Revenue Impact (Monte Carlo Simulation with Contagion)")
plt.show()


# -------------------------------- #
# Network Visual with Risk Gradient
# ---------------------------------#

# Normalize the adjusted risk scores to [0,1]
risk_values = np.array(list(adjusted_final_risk.values()))
min_risk = risk_values.min()
max_risk = risk_values.max()

norm_risk = {
    node: (adjusted_final_risk[node] - min_risk) / (max_risk - min_risk)
    if max_risk > min_risk else 0
    for node in adjusted_final_risk
}

# Use plt.get_cmap if you're already using plt
cmap = plt.get_cmap('viridis')


# Create a figure and axes explicitly.
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the network on the specified Axes.
nx.draw(G, ax=ax, with_labels=True,
        node_color=[norm_risk[node] for node in G.nodes()],
        cmap=cmap, font_weight='bold')

# Create a ScalarMappable for the colorbar using the same colormap and normalization.
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_risk, vmax=max_risk))
sm.set_array([])  # set_array can be empty; it's just for the colorbar mapping

# Add the colorbar to the figure, specifying the Axes to attach to.
fig.colorbar(sm, ax=ax, label='Normalized Network Risk')
ax.set_title("Network Visualization with Adjusted Risk Score Gradient")
plt.show()

# ------------------- #
#   Risk Data Table   #
# ------------------- #


risk_df = pd.DataFrame.from_dict(risk_details, orient='index')
risk_df.index.name = 'node'
risk_df.reset_index(inplace=True)


fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')


table = ax.table(cellText=risk_df.values,
                 colLabels=risk_df.columns,
                 cellLoc='center',
                 loc='center')


for key, cell in table.get_celld().items():
    try:
        cell.get_text().set_wrap(True)
        cell.set_fontsize(10) 
    except Exception as e:
        print(f"Warning: could not wrap text in cell {key}: {e}")

table.scale(1.1, 2)


plt.title("Risk Details Table", fontsize=16)
plt.show()

