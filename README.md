## Farabi Experiments
### Multi Agent Network Analysis 

This project is a framework for analyzing and simulating risks in a network of interconnected agents (nodes). Here's a simplified explanation of what it does:

Data Input: The project reads various data files that describe the network structure (adjacency matrix), error policies, costs, and additional risk factors like resource threats and SLA (Service Level Agreement) uptimes.

Network Construction: It builds a directed graph (network) using the adjacency matrix, where nodes represent agents and edges represent connections between them.

Centrality Measures: It calculates centrality measures (degree, betweenness, eigenvector, and closeness) for each node to understand their importance in the network.

Risk Calculation:

It computes a raw risk score for each node based on the centrality measures.
It adjusts these scores using error policies and additional risk factors like resource availability, design threats, and SLA compliance.
Simulation:

It runs a Monte Carlo simulation to model how risks propagate through the network over multiple rounds.
It simulates failures and their impact on the network's overall revenue.
Visualization:

It visualizes the network with nodes colored based on their risk levels.
It generates a histogram showing the distribution of network net revenue impacts from the simulation.
Reporting:

It prints detailed risk information for each node.
It displays a table summarizing the risk details.
In essence, the project helps to identify and quantify risks in a network of agents, simulate their potential impacts, and visualize the results for better understanding and decision-making.

#### Run any file with agent_network prefix 
```python3 agent_network_<whatever rest of filename>```
#### This file contains all the threat files in the MC sim
```python3 agent_network_simulation_contagion_with_all_threats.py``` 

#### definitions.txt has more details on specific risk components 

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

### Agentic pattern threat levels with centrality measures (added)
 Ken Huang's analysis was used to define the original design_threats.csv file
 https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro#

### Approximation for NP-hard loops
from sklearn.ensemble import RandomForestRegressor

 def approximate_risk(node):
    model = RandomForestRegressor()
    model.fit(historical_centralities, historical_errors)
    return model.predict(current_state)








