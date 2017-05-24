[![Build Status](https://travis-ci.org/MDAnalysis/pytng.svg?branch=master)](https://travis-ci.org/MDAnalysis/pytng.svg?branch=master)

# pytng

Python bindings for TNG file format

# Installation

We are not including a static copy of tng. Instead we dynamically load your
local installed library.

## Install TNG library

   If your package manager doesn't support it use it here
   
   ```bash
   cd <tng directory>
   mkdir -p build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX:PATH=$HOME/.local ..
   make install
   ```
   
   This will install tng in your home directory
   
   
## build pytng

   This will only work after installing tng

   ```bash
   CFLAGS="-I~/.local/include" LDFLAGS="-L~/.local/lib64" python setup.py develop
   ```


