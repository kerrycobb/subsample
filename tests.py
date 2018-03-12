import numpy as np
import networkx as nx
import subsample as ss
import pandas as pd

N = 10
n = 5
a = 1
b = 1
np.random.seed(seed=123)
x = np.random.beta(a=a, b=b, size=N)
y = np.random.beta(a=a, b=b, size=N)
coords = np.stack((x,y), axis=1)

G = ss.SpatialGraph(coords)

# subsample methods
Sg1 = ss.SubsampleGraph(G, n)
ss.plot_subsample(G, Sg1, show=False)

Sg2 = ss.SubsampleGraph(G, n, method='iter_drop_shortest')
ss.plot_subsample(G, Sg2, show=False)

Sg3 = ss.SubsampleGraph(G, n, method='max_sum_of_min_edges')
ss.plot_subsample(G, Sg3, show=False)

Sg4 = ss.SubsampleGraph(G, n, method='max_mean_area')
ss.plot_subsample(G, Sg4, show=False)

# Sg5 = ss.SubsampleGraph(G, n, method='max_sum_of_edges')
# ss.plot_subsample(G, Sg5, show=False)
#
# Sg6 = ss.SubsampleGraph(G, n, method='max_median_area')
# ss.plot_subsample(G, Sg6, show=False)
#
# Sg7 = ss.SubsampleGraph(G, n, method='max_area_sum')
# ss.plot_subsample(G, Sg7, show=False)
#
# Sg8 = ss.SubsampleGraph(G, n, method='min_area_var')
# ss.plot_subsample(G, Sg8, show=False)

# subsample
ix, sub_coords = ss.subsample(coords, n)

# subsample_df
df = pd.DataFrame(coords)
sub_df = ss.subsample_df(df, n, x=0, y=1)

print('Tests Successful!')
