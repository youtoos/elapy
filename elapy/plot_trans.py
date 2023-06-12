import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
sns.set_context('talk', font_scale=0.8)

def plot_trans(freq, trans, trans2):
  fig, (ax1, ax2, ax3) = plt.subplots(figsize=(5,10), nrows=3)

  # freq
  sns.barplot(data=freq.reset_index(), x='index', y='freq', ax=ax1,
              color=plt.cm.tab10(0))
  ax1.tick_params(length=5)
  ax1.margins(0.05, 0.2)
  ax1.set_xlabel('State number')
  ax1.set_ylabel('Frequency')

  # trans, trans2
  def func(data, ax, title):
    sns.heatmap(data=data, ax=ax, cmap='Blues', annot=True, square=True,
                xticklabels=1, yticklabels=1, cbar_kws=dict(aspect=10))
    ax.tick_params(length=0)
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_linewidth(1)
    cb = ax.collections[0].colorbar
    cb.outline.set_linewidth(1)
    cb.ax.tick_params(length=5, width=1)

  func(trans, ax2, 'Direct transitions')
  func(trans2, ax3, 'Direct/indirect transitions')
  fig.tight_layout()
  fig.show()
  fig.savefig('fig_trans.png')

if __name__ == '__main__':
  plot_trans(freq, trans, trans2)
