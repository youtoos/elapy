import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from .core import uniform_layout
sns.set_context('talk', font_scale=0.8)

def convert_orig(sr, n):
  return sr.apply(bin).str.split('b').str[1].str.rjust(n, '0').\
    str[::-1].apply(int, args=(2,))+1

def plot_basin_graph(graph_in, original_notation=False):
  graph = graph_in.copy()
  if original_notation:
    n = int(np.log2(len(graph)))
    graph.index = convert_orig(graph.index.to_series(), n)
    graph['source'] = convert_orig(graph.source, n)
    graph['target'] = convert_orig(graph.target, n)

  sr = graph.state_no.value_counts().sort_index()
  ncols = int(np.ceil(sr.size**0.5))
  nrows = int(np.ceil(sr.size / ncols))
  min_val = graph.energy.min()
  max_val = graph.energy.max()
  vmin = min_val - 0.2 * (max_val - min_val)
  vmax = max_val + 0.2 * (max_val - min_val)
  fig, axes = plt.subplots(figsize=(3*ncols, 3*nrows),
                           ncols=ncols, nrows=nrows)
  axes = axes.flatten()
  for (state_no, n_node), ax in zip(sr.items(), axes):
    df = graph[graph.state_no==state_no]
    G1 = nx.from_pandas_edgelist(df, create_using=nx.Graph)
    G2 = nx.from_pandas_edgelist(df, create_using=nx.DiGraph)
    pos = uniform_layout(G1, seed=0)
    node_size = 2000 / len(df)**0.5
    font_size = node_size**0.5 / 2
    nx.draw_networkx(G2, pos=pos, ax=ax,
                     node_size=node_size, font_size=font_size,
                     vmin=vmin, vmax=vmax, cmap='RdYlBu',
                     node_color=df.loc[list(G2.nodes)].energy,
                     linewidths=1, edgecolors='0.1',
                     edge_color='0.1', font_color='0.1')
    ax.text(0.05, 0.95, f'State {state_no}', transform=ax.transAxes)
    ax.margins(0.1, 0.2)
  for ax in axes:
    ax.axis('off')
  fig.suptitle('Basin graph')
  fig.tight_layout(pad=0, rect=[0,0,1,0.9])

  # color bar
  ax = fig.add_axes([0.8, 0.92, 0.12, 0.02])
  n_cb = 100
  sns.heatmap([min_val + (max_val-min_val) * np.arange(n_cb)/n_cb],
              cbar=False, cmap='RdYlBu', ax=ax, vmin=vmin, vmax=vmax)
  ax.set_yticks([])
  ax.set_xticks([0,n_cb])
  ax.set_xticklabels([f'{min_val:.3g}', f'{max_val:.3g}'], rotation=0)
  ax.tick_params(length=0)
  ax.set_title('Energy')

  fig.show()
  fig.savefig('fig_basin_graph.png')


if __name__ == '__main__':
  plot_basin_graph(graph)
