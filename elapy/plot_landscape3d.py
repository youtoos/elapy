import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import distance, linkage, leaves_list
import plotly.graph_objects as go

def _calc_pos_graph(Z, z_sr):

  # internal node number (index), child 1, and child 2
  n = len(Z) + 1
  info_df = pd.DataFrame(Z[:, [0,1]].astype(int),
                         index=range(n, 2*n-1),
                         columns=['c1', 'c2'])
  info_df = info_df.sort_index(ascending=False)

  # target position for each state
  z_ptp     = z_sr.max() - z_sr.min()
  theta_arr = 2 * np.pi * (np.argsort(leaves_list(Z)) + 1)/(n+1) + np.pi/2
  target_df = pd.DataFrame()
  target_df['x'] = 1.5 * z_ptp * np.cos(theta_arr)
  target_df['y'] = 1.5 * z_ptp * np.sin(theta_arr)

  # generate network
  G = nx.DiGraph()
  for i, (c1, c2) in info_df.iterrows():
    G.add_edge(i, c1)
    G.add_edge(i, c2)

  # calculate coordinates  ----------------------------------

  pos_df = pd.DataFrame(0.0, index=range(2*n-1), columns=list('xy'))
  pos_df.iloc[-1] = (0,0)

  l_min = 0.1 * z_ptp
  for i, (c1, c2) in info_df.iterrows():

    # edge length
    l1 = np.max([z_sr[i] - z_sr[c1], l_min])
    l2 = np.max([z_sr[i] - z_sr[c2], l_min])

    # averaged target position
    t1_arr = np.array(list(nx.descendants(G, c1)))
    t2_arr = np.array(list(nx.descendants(G, c2)))
    t1_arr = t1_arr[t1_arr < n] if len(t1_arr)>0 else np.array([c1])
    t2_arr = t2_arr[t2_arr < n] if len(t2_arr)>0 else np.array([c2])
    pos_t1 = target_df.loc[t1_arr].mean()
    pos_t2 = target_df.loc[t2_arr].mean()

    # edge direction
    pos_i = pos_df.loc[i]
    v1    = pos_t1 - pos_i
    v2    = pos_t2 - pos_i

    # set child positions
    pos_df.loc[c1] = pos_i + l1 * v1 / np.linalg.norm(v1)
    pos_df.loc[c2] = pos_i + l2 * v2 / np.linalg.norm(v2)

  # add z-coordinates
  pos_df['z'] = z_sr

  return pos_df, G


def plot_landscape3d(D_in):

  # hierarchical clustering
  D = D_in.copy()
  n = len(D)
  min_diag = np.diag(D).min()
  min_non_diag = D.values[np.triu_indices_from(D,1)].min()
  shift = 0.5 * min_diag + 0.5 * min_non_diag
  D -= shift
  np.fill_diagonal(D.values, 0)
  Z = linkage(distance.squareform(D))

  # calculate vertex positions and graph
  z_sr = pd.Series(np.concatenate([np.diag(D_in), Z[:,2] + shift]))
  pos_df, G = _calc_pos_graph(Z, z_sr)

  # calculate plot region
  margin = 0.5
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

  # graph
  n_div = 20
  out_list_x = []
  out_list_y = []
  for v1, v2 in G.edges:
    sr1 = pos_df.loc[v1]
    sr2 = pos_df.loc[v2]
    out_list_x.append(np.linspace(sr1.x, sr2.x, n_div))
    out_list_y.append(np.linspace(sr1.y, sr2.y, n_div))
  gx_arr = np.concatenate(out_list_x).reshape(-1, n_div)
  gy_arr = np.concatenate(out_list_y).reshape(-1, n_div)
  X = gx_arr.reshape(-1) - pos_df.x.values.reshape(-1,1)
  Y = gy_arr.reshape(-1) - pos_df.y.values.reshape(-1,1)
  w_df = pd.DataFrame(1/(X**2 + Y**2 + epsilon))
  w_df.loc[len(w_df)] = ambient_str * d**2 * np.ones(w_df.shape[1])
  w_df = w_df / w_df.sum()
  gz_arr = (w_df.T * z_arr).T.sum().values.reshape(-1, n_div)

  # create fig
  fig = go.Figure()

  # surcface plot
  n_line = 40
  contours = dict(x=dict(highlight=False, show=True, color='white',
                         start=x_min, end=x_max,
                         size=(x_max-x_min)/n_line),
                  y=dict(highlight=False, show=True, color='white',
                         start=y_min, end=y_max,
                         size=(y_max-y_min)/n_line),
                  z=dict(highlight=False, show=False))
  fig.add_trace(go.Surface(x = x_arr,
                           y = y_arr,
                           z = zz_arr,
                           colorscale = 'viridis',
                           hoverinfo = 'skip',
                           opacity = 1,
                           contours = contours))

  # graph
  dz = 0.005 * zz_arr.ptp()
  for gx_sub_arr, gy_sub_arr, gz_sub_arr in zip(gx_arr, gy_arr, gz_arr):
    fig.add_trace(go.Scatter3d(x = gx_sub_arr,
                               y = gy_sub_arr,
                               z = gz_sub_arr + dz,
                               mode = 'lines',
                               line = dict(width=5, color='white')))

  # state number
  dz = 0.02 * zz_arr.ptp()
  pos_sub_df = pos_df.iloc[:n]
  fig.add_trace(go.Scatter3d(x = pos_sub_df.x,
                             y = pos_sub_df.y,
                             z = pos_sub_df.z + dz,
                             text = pos_sub_df.index + 1,
                             mode = 'markers+text',
                             hoverinfo = 'skip',
                             marker = dict(size=4, color='white'),
                             textfont = dict(size=20, color='white'),
                             textposition = 'top center'))


  # other setting
  fig.update_layout(showlegend=False)
  axis = dict(visible=False)
  fig.update_scenes(xaxis=axis, yaxis=axis, zaxis=axis,
                    aspectratio=dict(x=1.5, y=1.5, z=0.2),
                    camera_eye=dict(x=0, y=-1.25, z=1.25))

  # show
  fig.write_html('tmp.html')
  if 'TerminalIPythonApp' not in get_ipython().config:
    fig.show()

if __name__ == '__main__':
  plot_landscape3d(D)
