import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
sns.set_context('talk', font_scale=0.8)

def plot_basin_graph(graph):
  sr = graph.basin_id.value_counts()
  ncols = int(np.ceil(sr.size**0.5))
  nrows = int(np.ceil(sr.size / ncols))
  vmin = graph.energy.min()
  vmax = graph.energy.max()
  vmin -= 0.1 * (vmax - vmin)
  vmax += 0.1 * (vmax - vmin)
  fig, axes = plt.subplots(figsize=(3*ncols, 3*nrows),
                           ncols=ncols, nrows=nrows)
  axes = axes.flatten()
  for (basin_id, n_node), ax in zip(sr.iteritems(), axes):
    df = graph[graph.basin_id==basin_id]
    G1 = nx.from_pandas_edgelist(df, create_using=nx.Graph)
    G2 = nx.from_pandas_edgelist(df, create_using=nx.DiGraph)
    pos = nx.spring_layout(G1, seed=0, k=1/df.size**0.2)
    node_size = 2000 / len(df)**0.5
    font_size = 50 / len(df)**0.5
    nx.draw_networkx(G2, pos=pos, ax=ax,
                     node_size=node_size, font_size=font_size,
                     vmin=vmin, vmax=vmax, cmap='RdYlGn',
                     node_color=df.loc[list(G2.nodes)].energy,
                     edge_color='0.1', font_color='0.1')
    ax.margins(0.1)
  for ax in axes:
    ax.axis('off')
  fig.tight_layout(pad=0)
  fig.show()


if __name__ == '__main__':
  plot_basin_graph(graph)
