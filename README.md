[![Build Status](https://travis-ci.org/MDAnalysis/pytng.svg?branch=master)](https://travis-ci.org/MDAnalysis/pytng)
[![codecov](https://codecov.io/gh/MDAnalysis/pytng/branch/master/graph/badge.svg)](https://codecov.io/gh/MDAnalysis/pytng)

# pytng

Python bindings for TNG file format


```python
import pytng

with pytng.TNGFile('traj.tng', 'r') as f:
    for ts in f:
        coordinates = ts.positions
```

