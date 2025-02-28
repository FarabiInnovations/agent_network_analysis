## Farabi Experiments
### Multi Agent Network Analysis 

The framework serves as a working baseline for multi-agent risk analysis, providing scaffolding for exploratory analyses. 

#### Run any file with agent_network prefix 
```python3 agent_network_<filename>```
#### This file contains all the threat files in the MC contagion sim
```python3 agent_network_simulation_contagion_with_all_threats.py``` 

#### More details on risk components 

[definitions.md](./definitions.md)

#### Some strengths:

Cross-Domain Risk Transferability --
Foundational financial network principles adapted to AI contexts through eigenvector and betweenness centrality interpretations.

Actionable Risk Prioritization --
Combined topological-economic scoring enables targeted hardening of critical nodes 

Monte Carlo Foundation--
Three-stage simulation architecture (independent → dependent → stable) provides expandable structure.

This open-source framework is released under the MIT License, and the code is available for further experimentation and refinement.

#### Some limitations and a few next steps:

Temporal Dynamics: Add rolling adjacency matrices with reinforcement learning (RL) for more of an adaptive risk mitigation.

Security Integration: Security policy. Map the AI-specific attack vectors to centrality metrics (Done). 

Validation: Run some of the benchmarking against hybrid network topologies with some adversarial node injections - 


### For weighted adjacency matrix

A weighted adjacency matrix shows the strength or intensity of the connection. For traffic flow, each entry can represent the proportion of traffic that moves between nodes, or how likely a node is to serve as a fallback if the primary route isn’t available. 

Each node might show a self-connection value of 1.0. This could be interpreted as the baseline 
traffic that stays at the node (or a normalization factor). Basically that the majority of traffic remains at its origin unless rerouted. 

Off diagonal values, in the case of A and B example, they might 
have a mutual connection, such that if node A were to have a disruption in processing its own traffic, 
some % of As traffic might be diverted to B (and vice versa), in a tightly coupled pattern. The amount of traffic diverted would then be the weighted edge, i.e. if the weight were 0.2 from A to B, that might indicate 20% of As traffic gets diverted to B when A has problems. 

A high fallback percentage after transformation, becomes a low distance. This 
should make that edge more attractive for shortest path routing. This is the idea 
that edges with high fallback traffic are more preferred paths.

Example weighted adjacency matrix: network_centrality > agent_network_weighted_centrality.py

### Undirected vs. directed graph ###
This baseline uses a mixed graph of both directed and undirected edges, information can flow in one direction
or bi-directionally. If a model desires that every connection is assumed bi-directional, then this method can be applied:

G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())\

G = G.to_undirected() 

### Improvement ideas ###

#### Temporal Network Extension, i.e. time-varying adjacency matrices
something like;\
class TemporalNetwork:
    def __init__(self, time_slices):
        self.slices = [nx.from_pandas_adjacency(df) for df in time_slices]
        
    def rolling_centrality(self, window_size=5):
        return [compute_centrality(slice) for slice in self.slices[-window_size:]]

#### Agentic pattern threat levels with centrality measures (added)
 Ken Huang's analysis was used to define the original design_threats.csv file

 [Ken Huang's analysis](https://cloudsecurityalliance.org/blog/2025/02/06/agentic-ai-threat-modeling-framework-maestro#)


#### Approximation for NP-hard loops
from sklearn.ensemble import RandomForestRegressor

 def approximate_risk(node):
    model = RandomForestRegressor()
    model.fit(historical_centralities, historical_errors)
    return model.predict(current_state)








