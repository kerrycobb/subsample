# Subsample

Compute or approximate the most uniformly distributed subset of a given set of points


## Installation

```bash
pip install git+https://github.com/kerrycobb/subsample
```

## Usage

```python

import numpy as np
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

ix, sub_coords = ss.subsample(coords, n)

df = pd.DataFrame(coords)
sub_df = ss.subsample_df(df, n, x=0, y=1)

```
