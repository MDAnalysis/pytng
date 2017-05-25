import pytng

import numpy as np
import pytest


def test_n_molecules(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert tng.n_molecules == GMX_REF_DATA.n_molecules
        tng.close()
        with pytest.raises(IOError):
            assert tng.n_molecules == GMX_REF_DATA.n_molecules


def test_n_atomtypes(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        types = tng.atomtypes
        assert np.array_equal(types, ['O', 'H', 'H'] * 5)


def test_n_atomnames(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        names = tng.atomnames
        print(names)
        assert np.array_equal(names, ['O', 'HO1', 'HO2'] * 5)
