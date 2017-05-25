import pytng

import numpy as np
import pytest


def test_n_molecules(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert tng.n_molecules == GMX_REF_DATA.n_molecules


def test_n_molecules_closed_IOError(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.close()
        with pytest.raises(IOError):
            _ = tng.n_molecules


def test_atomtypes(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        types = tng.atomtypes
        assert np.array_equal(types, [b'O', b'H', b'H'] * 5)


def test_atomtypes_closed_IOError(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.close()
        with pytest.raises(IOError):
            _ = tng.atomtypes


def test_atomnames(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        names = tng.atomnames
        assert np.array_equal(names, [b'O', b'HO1', b'HO2'] * 5)


def test_atomnames_closed_IOError(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.close()
        with pytest.raises(IOError):
            _ = tng.atomnames


def test_chainnames(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        names = tng.chainnames
        assert np.array_equal(names, [b'W', ] * 15)


def test_chainnames_closed_IOError(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.close()
        with pytest.raises(IOError):
            _ = tng.chainnames


def test_n_chains(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert 1 == tng.n_chains


def test_n_residues(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert 5 == tng.n_residues


def test_residue_ids(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        ids = tng.residue_ids
        assert np.array_equal(ids,
                              [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])


def test_residue_ids_closed_IOError(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.close()
        with pytest.raises(IOError):
            _ = tng.residue_ids


def test_residue_names(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        ids = tng.residue_names
        assert np.array_equal(ids, [b'WAT', ] * 15)


def test_residue_names_closed_IOError(GMX_REF_FILEPATH):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        tng.close()
        with pytest.raises(IOError):
            _ = tng.residue_names
