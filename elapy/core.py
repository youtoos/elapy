import numpy as np
import pandas as pd
import networkx as nx

# Notation
# W: weight matrix
# h: coefficient vector
# n: number of variables
# k: number of observations
# m: number of all states, 2**n

def load_testdata(n=1):
  data_file_name = f'test_data/testdata_{n}.dat'
  roi_file_name = 'test_data/roiname.dat'
  X = pd.read_table(data_file_name, header=None)
  X.index = pd.read_csv(roi_file_name, header=None).squeeze()
  return (X==1).astype(int)

def binarize(X):
  return ((X.T - X.T.mean()).T >=0).astype(int)

def calc_state_no(X):
  return X.astype(str).sum().apply(lambda x:int(x,base=2))

def gen_all_state(X_in):
    n = len(X_in)
    X =  np.array([list(bin(i)[2:].rjust(n,'0'))
                for i in range(2**n)]).astype(int).T
    return pd.DataFrame(X, index=X_in.index)

def calc_energy(h, W, X_in):
  X = 2*X_in-1
  return -0.5 * (X * W.dot(X)).sum() - h.dot(X)

def calc_prob(h, W, X):
  energy  = calc_energy(h, W, X)
  energy -= energy.min()  # avoid overflow
  prob    = np.exp(-energy)
  return prob / prob.sum()

# pseudo-likelihood
def fit_approx(X_in, max_iter=10**3, alpha=0.9):
  X      = 2*X_in-1
  n, k   = X.shape
  h      = np.zeros(n)
  W      = np.zeros((n,n))
  X_mean = X.mean(axis=1)
  X_corr = X.dot(X.T) / k
  np.fill_diagonal(X_corr.values, 0)
  for i in range(max_iter):
    Y  = np.tanh(W.dot(X).T + h) # k * n
    h += alpha * (X_mean - Y.mean(axis=0))
    Z  = X.dot(Y) / k
    Z  = (Z + Z.T) / 2
    np.fill_diagonal(Z.values, 0)
    W += alpha * (X_corr - Z)
    if np.allclose(X_mean, Y.mean(axis=0)) and np.allclose(X_corr, Z):
      break
  return h, W

# likelihood
def fit_exact(X_in, max_iter=10**4, alpha=0.5):
  X      = 2*X_in-1
  X_all  = gen_all_state(X)
  X2_all = 2*X_all-1
  n, k   = X.shape
  m      = 2**n
  h      = np.zeros(n)
  W      = np.zeros((n,n))
  X_mean = X.mean(axis=1)
  X_corr = X.dot(X.T) / k
  np.fill_diagonal(X_corr.values, 0)
  for i in range(max_iter):
    p      = calc_prob(h, W, X_all)
    Y_mean = X2_all.dot(p)
    Y_corr = X2_all.dot(np.diag(p)).dot(X2_all.T)
    np.fill_diagonal(Y_corr.values, 0)
    h += alpha * (X_mean - Y_mean)
    W += alpha * (X_corr - Y_corr)
    if np.allclose(X_mean, Y_mean) and np.allclose(X_corr, Y_corr):
      break
  return h, W

def calc_accuracy(h, W, X):
  freq  = calc_state_no(X).value_counts()
  p_n   = freq / freq.sum()
  q     = X.mean(axis=1)
  X_all = gen_all_state(X)
  p_1   = (X_all.T * q + (1-X_all).T * (1-q)).T.prod()
  p_2   = calc_prob(h, W, X_all)
  def entropy(p):
    return (-p * np.log2(p)).sum()
  r  = (entropy(p_1) - entropy(p_2)) / (entropy(p_1) - entropy(p_n))
  d1 = (p_n * np.log2(p_n / p_1.iloc[p_n.index])).sum()
  d2 = (p_n * np.log2(p_n / p_2.iloc[p_n.index])).sum()
  rd = (d1-d2)/d1
  print(r, rd)

def calc_adjacent(X):
  X_all = gen_all_state(X)
  out_list = [calc_state_no(X_all)]
  for i in X_all.index:
    Y = X_all.copy()
    Y.loc[i] = 1 - Y.loc[i]
    out_list.append(calc_state_no(Y))
  return pd.concat(out_list, axis=1)

def calc_basin_graph(h, W, X):
  X_all = gen_all_state(X)
  A = calc_adjacent(X)
  energy = calc_energy(h, W, X_all)
  min_idx = energy.values[A].argmin(axis=1)
  graph = pd.DataFrame()
  graph['source'] = A.index.values
  graph['target'] = A.values[A.index, min_idx]
  graph['energy'] = energy
  G = nx.from_pandas_edgelist(graph, create_using=nx.DiGraph)
  graph['basin_id'] = 0
  for i, node_set in enumerate(nx.weakly_connected_components(G)):
    graph.loc[list(node_set),'basin_id'] = i
  return graph

def calc_trans(X, graph):
  sr = graph.loc[calc_state_no(X)].basin_id
  freq = sr.value_counts().sort_index()
  sr = sr[sr.diff()!=0]  # change points only
  trans = pd.crosstab(sr.values[:-1], sr.values[1:])
  trans.index.name ='src'
  trans.columns.name ='dst'
  out_list = []
  for i in freq.index:
    for j in freq.index:
      sr2 = sr[sr.isin([i,j])]
      sr2 = sr2[sr2.diff()!=0]
      count = int(sr2.size / 2)
      out_list.append(dict(src=i, dst=j, count=count))
  trans2 = pd.DataFrame(out_list)  # includes indirect transitions
  trans2 = trans2.set_index(['src','dst'])['count'].unstack()
  np.fill_diagonal(trans2.values, 0)
  return freq, trans, trans2

def calc_discon_graph_sub(i_input, H, A):
  m, n = A.shape
  C = np.inf * np.ones(m)
  C[i_input] = H[i_input]
  I = set(range(m))
  while I:
    I_list = list(I)
    i = I_list[np.argmin(C[I_list])]
    I.remove(i)
    for j in A[i]:
      if j in I:
        if C[i] <= H[j]:
          C[j] = H[j]
        else:
          C[j] = min(C[i], C[j])
  return C

def calc_discon_graph(h, W, X, graph):
  X_all = gen_all_state(X)
  A = calc_adjacent(X).values[:, 1:]  # remove self-loop
  H = calc_energy(h, W, X_all).values
  local_idx = graph[graph.source==graph.target].index
  out_list = []
  for i_input in local_idx:
    C = calc_discon_graph_sub(i_input, H, A)
    out_list.append(C[local_idx])
  D = pd.DataFrame(np.array(out_list), index=local_idx, columns=local_idx)
  return D
