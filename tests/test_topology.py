import pytng

import numpy as np
import pytest


def test_n_molecules(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert tng.n_molecules == GMX_REF_DATA.n_molecules
        tng.close()
        with pytest.raises(IOError):
            assert tng.n_molecules == GMX_REF_DATA.n_molecules


def test_atomtypes(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        types = tng.atomtypes
        assert np.array_equal(types, ['O', 'H', 'H'] * 5)


def test_atomnames(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        names = tng.atomnames
        assert np.array_equal(names, ['O', 'HO1', 'HO2'] * 5)


def test_chainnames(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        names = tng.chainnames
        assert np.array_equal(names, ['W', ] * 15)


def test_residue_ids(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        ids = tng.residue_ids
        assert np.array_equal(ids,
                              [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
