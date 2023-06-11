# elapy
This is a Python implementation of Energy Landscape Analysis Toolkit (ELAT). An Ising model is fit to the input data that should be a {0,1}-valued or {-1,1}-valued matrix. From the estimated Ising model, local minima, basins of attractions, and a disconnectivity graph that shows the minimum values of the maximum energy of intermediate states passed when moving between local minima are calculated. For more details, please see the original repository or the original paper shown below.

The original Matlab codes written by Dr. T. Ezaki are available at: https://github.com/tkEzaki/energy-landscape-analysis.

The original paper (T. Ezaki, et al., Philos. Trans. A Math. Phys. Eng. Sci., 2017) is available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5434078/. 

Japanese documents are also avaiable at the original codes' author's website: https://sites.google.com/site/ezakitakahiro/software.

# Required packages
numpy, scipy, pandas, matplotlib, seaborn, and networkx

# Install
```
git clone https://github.com/okumakito/elapy.git
```

# Usage

* Import
  ```
  import elapy as ela
  ```
* Convert your data to a {0,1}-valued pandas DataFrame object with rows represents variables and columns represents observations or time points. You can load a test data provided by the original repository (https://github.com/tkEzaki/energy-landscape-analysis) by the following command:
  ```
  data = ela.load_testdata(1)  # the argument is 1, 2, 3, or 4
  ```
* Fit the data to an Ising model. Please choose one from the two functions below depending on the data size.

  ```
  h, W = ela.fit_exact(data)  # exact fitting based on likelihood function
  h, W = ela.fit_approx(data) # approximated fitting based on pseudo-likelihood function
  ```
* Calculate a basin graph.
  ```
  graph = ela.calc_basin_graph(h, W, data)
  ```
* Calculate a disconnectivity graph.
  ```
  D = ela.calc_discon_graph(h, W, data, graph)
  ```
* Plot figures
  ```
  ela.plot_local_min(data, graph)
  ela.plot_basin_graph(graph)
  ela.plot_discon_graph(D)
  ```
 
