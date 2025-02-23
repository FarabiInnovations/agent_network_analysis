import numpy as np
import networkx as nx

# Existing caluculations can be used, here is stub data example
# final_risk_scores = {'A': 0.15, 'B': 0.35, 'C': 0.55, 'D': 0.70, 'E': 0.25}
# G is built NetworkX DiGraph.

# stub data
final_risk_scores = {'A': 0.15, 'B': 0.35, 'C': 0.55, 'D': 0.70, 'E': 0.25}

# if a node's risk exceeds this then reroute attempt.
RISK_THRESHOLD = 0.1

def adaptive_routing_with_risk(start_node, max_hops=5):
    """
    Processes a request starting from 'start_node'. 
    If the risk score at the current node exceeds RISK_THRESHOLD,
    the request is rerouted to the neighbor with the lowest risk.
    The process stops when a node with acceptable risk is reached or
    after max_hops.
    Returns the path taken.
    """
    path = [start_node]
    current_node = start_node

    for _ in range(max_hops):
        # if within acceptable limit, process request
        if final_risk_scores[current_node] <= RISK_THRESHOLD:
            break

        # otherwise, find neighbors and select the one with the lowest risk score
        neighbors = list(G.successors(current_node))
        if not neighbors:
            # no fallback available
            break

        next_node = min(neighbors, key=lambda n: final_risk_scores.get(n, 1))
        path.append(next_node)
        current_node = next_node

    return path

# example; simulate requests entering through example user-facing nodes A or B
## in practice user facing nodes can be dynamically captured using adjanceny matrix 'self-loop'
## or more robustly as nodes defined for application input where a flag in the error
## policy or some other policy (or some other method) indicates user facing  
start_nodes = ["A", "B"]
num_requests = 50
for i in range(num_requests):
    start = np.random.choice(start_nodes)
    route = adaptive_routing_with_risk(start)
    print(f"Request {i+1}: {' -> '.join(route)}")
