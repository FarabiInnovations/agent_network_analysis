import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# adjacency matrix 
csv_file = "../adjacency_matrices/weighted_adjacency_matrix.csv"  
adj_matrix = pd.read_csv(csv_file, index_col=0)

# convert the dataframe 
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())

# -----------------------------------------------------------------------------------------#
# Modelling fallbacks using the weights
# -----------------------------------------------------------------------------------------#
# If considering fallbacks. A high fallback percentage after transformation, 
# becomes a low distance. This should make that edge more attractive for shortest path routing.
# This is the idea that edges with high fallback traffic are more preferred paths.

for u, v, data in G.edges(data=True): fallback_weight = data.get('weight', 1)

# using inverse transformation:
data['distance'] = 1 / fallback_weight if fallback_weight > 0 else float('inf')

# optional log transform
# data['distance'] = -math.log(fallback_weight) if fallback_weight > 0 else float('inf')

degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G,weight='weight')
closeness_centrality = nx.closeness_centrality(G, distance='distance')
betweenness_centrality = nx.betweenness_centrality(G, weight='distance')

# -----------------------------------------------------------------------------------------#


# -----------------------------------------------------------------------------------------#
# Not using weights for fallback approach
# -----------------------------------------------------------------------------------------#

# centrality measures without distance considerations 
# degree_centrality = nx.degree_centrality(G)
# betweenness_centrality = nx.betweenness_centrality(G,weight='weight')
# eigenvector_centrality = nx.eigenvector_centrality(G,weight='weight')
# closeness_centrality = nx.closeness_centrality(G,distance='weight')

# -----------------------------------------------------------------------------------------#

centrality_metrics_df = pd.DataFrame({
    "Degree Centrality": degree_centrality,
    "Betweenness Centrality": betweenness_centrality,
    "Eigenvector Centrality": eigenvector_centrality,
    "Closeness Centrality": closeness_centrality
}).round(4) 


fig, ax = plt.subplots(figsize=(8, 4))

# Hide the axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# create a table
table = ax.table(cellText=centrality_metrics_df.values, 
                 colLabels=centrality_metrics_df.columns,
                 rowLabels=centrality_metrics_df.index, 
                 cellLoc="center", loc="center")

# table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2) 

# plots
plt.title("Network Centrality Metrics", fontsize=12, fontweight="bold")
plt.show()
