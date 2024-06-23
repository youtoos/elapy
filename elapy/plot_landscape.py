import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import distance, linkage
sns.set_context('talk', font_scale=0.8)

def branching_embedding(Z, z_sr, theta):

  # internal node number (index), child 1, and child 2
  n = len(Z) + 1
  info_df = pd.DataFrame(Z[:, [0,1]].astype(int),
                         index=range(n, 2*n-1),
                         columns=['c1', 'c2'])
  info_df = info_df.sort_index(ascending=False)

  # parent
  parent_sr = pd.Series(0, index=range(2*n-2))
  for i, (c1, c2) in info_df.iterrows():
    parent_sr[c1] = i
    parent_sr[c2] = i

  # calculate coordinates  ----------------------------------

  pos_df = pd.DataFrame(0.0, index=range(2*n-1), columns=list('xy'))
  pos_df.iloc[-1] = (0,0)

  l_min = 0.1 * (z_sr.max() - z_sr.min())
  for i, (c1, c2) in info_df.iterrows():
    l1 = np.max([z_sr[i] - z_sr[c1], l_min])
    l2 = np.max([z_sr[i] - z_sr[c2], l_min])

    if i == 2*n - 2:
      # first branching
      th1  = theta / 2
      th2  = theta / 2
      phi  = -np.pi/2
      psi1 = phi + th1
      psi2 = phi - th2
      pos_df.loc[c1] = [l1 * np.cos(psi1), l1 * np.sin(psi1)]
      pos_df.loc[c2] = [l2 * np.cos(psi2), l2 * np.sin(psi2)]
    else:
      # second or later branching
      pos_p = pos_df.loc[parent_sr[i]]
      pos_i = pos_df.loc[i]
      th1   = np.arctan((l1 - l2 * np.cos(theta)) / (l2 * np.sin(theta)))
      th2   = theta - th1
      phi   = np.angle(np.complex(*(pos_i - pos_p)))
      psi1  = phi + th1
      psi2  = phi - th2
      pos_df.loc[c1] = pos_i + [l1 * np.cos(psi1), l1 * np.sin(psi1)]
      pos_df.loc[c2] = pos_i + [l2 * np.cos(psi2), l2 * np.sin(psi2)]

  # add z-coordinates
  pos_df['z'] = z_sr

  # generate network
  G = nx.DiGraph()
  for i, (c1, c2) in info_df.iterrows():
    G.add_edge(i, c1)
    G.add_edge(i, c2)

  return pos_df, G


def plot_landscape(D_in, theta=5*np.pi/6):

  # hierarchical clustering
  D = D_in.copy()
  n = len(D)
  min_diag = np.diag(D).min()
  min_non_diag = D.values[np.triu_indices_from(D,1)].min()
  shift = 0.5 * min_diag + 0.5 * min_non_diag
  D -= shift
  np.fill_diagonal(D.values, 0)
  Z = linkage(distance.squareform(D))

  # calculate xy-coordinates by brancing embedding
  z_sr = pd.Series(np.concatenate([np.diag(D_in), Z[:,2] + shift]))
  pos_df, G = branching_embedding(Z, z_sr, theta)

  # calculate plot region
  margin = 0.3
  x_min, x_max = pos_df.x.min(), pos_df.x.max()
  y_min, y_max = pos_df.y.min(), pos_df.y.max()
  d = (1 + margin) *  np.max([x_max - x_min, y_max - y_min])
  x_min = (x_min + x_max)/2 - d/2
  x_max = x_min + d
  y_min = (y_min + y_max)/2 - d/2
  y_max = y_min + d

  # generate mesh
  n_mesh = 100
  x_arr = np.linspace(x_min, x_max, n_mesh)
  y_arr = np.linspace(y_min, y_max, n_mesh)
  xx_arr, yy_arr = np.meshgrid(x_arr, y_arr)

  # interpolate z-coordinates
  epsilon = 1e-5
  ambient_str = 0.5
  X = xx_arr.reshape(-1) - pos_df.x.values.reshape(-1,1)
  Y = yy_arr.reshape(-1) - pos_df.y.values.reshape(-1,1)
  w_df = pd.DataFrame(1/(X**2 + Y**2 + epsilon))
  w_df.loc[len(w_df)] = ambient_str * d**2 * np.ones(w_df.shape[1])
  w_df = w_df / w_df.sum()
  z_arr = np.concatenate([pos_df.z.values, [pos_df.z.max()]])
  zz_arr = (w_df.T * z_arr).T.sum().values.reshape(n_mesh, n_mesh)

  # prepare fig and axes
  fig, ax = plt.subplots(figsize=(6,5))

  # contour plot
  g = ax.contourf(xx_arr, yy_arr, zz_arr, levels=20, cmap='viridis')

  # color bar
  cbar = fig.colorbar(g, shrink=0.6, aspect=10)
  cbar.outline.set_linewidth(1)
  cbar.ax.tick_params(length=2, width=1)

  # graph
  pos_dict = {i:[x,y] for i, (x,y,z) in pos_df.iterrows()}
  nx.draw_networkx(G,
                   pos        = pos_dict,
                   node_size  = 30,
                   node_color = 'w',
                   edgecolors = 'w', # node border color
                   linewidths = 0,   # node border width
                   edge_color = 'w', # edge color
                   width      = 1,   # edge width
                   font_size  = 0,
                   ax = ax)

  # state number
  pad = 0.04 * d
  for i, sr in pos_df.iloc[:n].iterrows():
    parent = list(G.predecessors(i))[0]
    sr2 = pos_df.loc[parent]
    phi = np.angle(np.complex(*([sr.x - sr2.x, sr.y- sr2.y])))
    phi += np.pi/2
    ax.text(sr.x + pad * np.cos(phi), sr.y + pad * np.sin(phi), str(i+1),
            ha='center', va='center', color='w')

  # other setting
  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_title('Energy landscape', fontsize=16, pad=10)
  for spine in ax.spines.values():
    spine.set_linewidth(1)

  # show
  fig.tight_layout()
  fig.show()
  fig.savefig('fig_landscape.png')

if __name__ == '__main__':
  plot_landscape(D)
