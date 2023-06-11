import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)

def plot_local_min(data, graph):
  idx = graph[graph.source == graph.target].index
  n = len(data)
  X =  np.array([list(bin(i)[2:].rjust(n,'0'))
                 for i in idx]).astype(int).T
  df = pd.DataFrame(X, index=data.index)
  df.columns = 'state' + (df.columns + 1).astype(str)

  fig, ax = plt.subplots(figsize=(4,4))
  sns.heatmap(data=df, ax=ax, linecolor='w', lw=2, square=True,
              cmap=sns.color_palette('Paired', 2),
              cbar_kws=dict(ticks=[0.25,0.75], shrink=0.25, aspect=2))
  ax.tick_params(length=0)
  ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
  ax.set_title('Local Minima', fontsize=16, pad=10)
  cax = ax.collections[0].colorbar.ax
  cax.set_yticklabels(['0','1'])
  cax.tick_params(length=0)
  fig.tight_layout()
  fig.show()

if __name__ == '__main__':
  plot_local_min(data, graph)
