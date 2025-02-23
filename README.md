## Farabi Experiments
### Multi Agent Network Analysis 

The framework serves as a working baseline for multi-agent risk analysis, providing scaffolding for exploratory analyses. With further refinement, it can evolve into a hardened risk assessment tool. 

#### Some strengths:

Cross-Domain Risk Transferability --
Foundational financial network principles adapted to AI contexts through eigenvector and betweenness centrality interpretations.

Actionable Risk Prioritization --
Combined topological-economic scoring enables targeted hardening of critical nodes 

Monte Carlo Foundation--
Three-stage simulation architecture (independent → dependent → stable) provides expandable structure.

#### Some limitations and a few next steps:

Temporal Dynamics: Add rolling adjacency matrices with reinforcement learning (RL) for more of an adaptive risk mitigation.

Security Integration: Security policy. Map the AI-specific attack vectors to centrality metrics. 

Validation: Run some of the benchmarking against hybrid network topologies with some adversarial node injections - 

This open-source framework is released under the MIT License, and the code is available for further experimentation and refinement.

### For weighted adjacency matrix
1. Update CSV with desired weighted edges
2. Add the weight='weight' parameter to the appropriate NetworkX centrality functions, 
e.g. betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

## Improvement ideas ##

### Temporal Network Extension, i.e. time-varying adjacency matrices
class TemporalNetwork:
    def __init__(self, time_slices):
        self.slices = [nx.from_pandas_adjacency(df) for df in time_slices]
        
    def rolling_centrality(self, window_size=5):
        return [compute_centrality(slice) for slice in self.slices[-window_size:]]

### Security risk factors with centrality measures
## Ken Huang's analysis was used to define the original design_threats.csv file
# https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro#
# agentic pattern threat levels can be layered into the centrality at the node (agent) level
security_risk = {
    node: betweenness_centrality[node] * injection_surface[node]
    for node in G.nodes()
}

final_risk_scores[node] += security_risk[node] * threat_intelligence[node]

### Approximation for NP-hard loops
from sklearn.ensemble import RandomForestRegressor

 def approximate_risk(node):
    model = RandomForestRegressor()
    model.fit(historical_centralities, historical_errors)
    return model.predict(current_state)








