import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import distance, linkage, dendrogram
sns.set_context('talk', font_scale=0.8)

def plot_discon_graph(D_in):
  D = D_in.copy()
  min_val = np.triu(D, 1).min()
  D -= min_val
  np.fill_diagonal(D.values, 0)
  Z = linkage(distance.squareform(D))
  dendro = dendrogram(Z, no_plot=True)
  D = D_in.iloc[dendro['leaves'], dendro['leaves']]

  fig, ax = plt.subplots(figsize=(6,4))
  color = plt.cm.tab10(0)

  # dendrogram above min_val
  for x_list, y_list in zip(dendro['icoord'], dendro['dcoord']):
    x_arr = (np.array(x_list) - 5) / 10
    y_arr = np.array(y_list) + min_val
    ax.plot(x_arr, y_arr, c=color)

  # dendrogram below min_val
  for i, energy in enumerate(np.diag(D)):
    ax.plot([i,i], [energy, min_val], c=color)

  ax.set_xticks(np.arange(len(D)))
  ax.set_xticklabels(D.index)
  ax.tick_params(length=5)
  ax.margins(0.1)
  ax.set_xlabel('State number')
  ax.set_ylabel('Energy')
  ax.set_title('Disconnectivity graph', fontsize=16, pad=10)
  sns.despine()
  fig.tight_layout()
  fig.show()
  fig.savefig('fig_discon_graph.png')

if __name__ == '__main__':
  plot_discon_graph(D)
