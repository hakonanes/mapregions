mapregions
----------

This Python package attempts to vectorize parts of scikit-image's `RegionProperties`. It
was initially made to store and manipulate particles detected in electron micrographs.

The package might become a module in the `orix Python package
<https://orix.readthedocs.io>`_, where it could be used to segment a `CrystalMap` into
regions (grains, particles, or similar).

`mapregions` is released under the MIT license.

Install from source::

    git clone https://github.com/hakonanes/mapregions
    cd mapregions
    pip install -e .
