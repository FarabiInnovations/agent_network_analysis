import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


csv_file = "../adjacency_matrices/example_adjacency_matrix.csv"   
adj_matrix = pd.read_csv(csv_file, index_col=0)  

# graph build
G = nx.from_pandas_adjacency(adj_matrix, create_using=nx.DiGraph())


degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)
closeness_centrality = nx.closeness_centrality(G)


centrality_metrics_df = pd.DataFrame({
    "Degree Centrality": degree_centrality,
    "Betweenness Centrality": betweenness_centrality,
    "Eigenvector Centrality": eigenvector_centrality,
    "Closeness Centrality": closeness_centrality
}).round(4)  

# figure
fig, ax = plt.subplots(figsize=(8, 4))


ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)


table = ax.table(cellText=centrality_metrics_df.values, 
                 colLabels=centrality_metrics_df.columns,
                 rowLabels=centrality_metrics_df.index, 
                 cellLoc="center", loc="center")


table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  

##plotting
plt.title("Network Centrality Metrics", fontsize=12, fontweight="bold")


plt.show()
