## Farabi Experiments
### Multi Agent Network Analysis 

The framework serves as a working baseline for multi-agent risk analysis, providing scaffolding for exploratory analyses. 

#### Run any file with agent_network prefix 
```python3 agent_network_<whatever rest of filename>```
#### This file contains all the threat files in the MC contagion sim
```python3 agent_network_simulation_contagion_with_all_threats.py``` 

#### More details on risk components 

definitions.md 

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

##### Transform the 'weight' attribute into a 'distance' attribute.
for u, v, data in G.edges(data=True):
    fallback_weight = data.get('weight', 1)\
     #####Using inverse transformation:\
     data['distance'] = 1 / fallback_weight if fallback_weight > 0 else float('inf')
    
    ##### optional log transform
    data['distance'] = -math.log(fallback_weight) if fallback_weight > 0 else float('inf')

1. Update CSV with desired weighted edges
2. Add the weight='weight' parameter to the appropriate NetworkX centrality functions
\
                degree_centrality = nx.degree_centrality(G) # no weight\
                betweenness_centrality = nx.betweenness_centrality(G,weight='weight')\
                eigenvector_centrality = nx.eigenvector_centrality(G,weight='weight')\
                closeness_centrality = nx.closeness_centrality(G,distance='weight')\
        ##### Use the 'distance' attribute for centrality calculations given fallback strategy\
                closeness_centrality = nx.closeness_centrality(G, distance='distance')\
                betweenness_centrality = nx.betweenness_centrality(G, weight='distance')

## Improvement ideas ##

### Temporal Network Extension, i.e. time-varying adjacency matrices
something like;\
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








