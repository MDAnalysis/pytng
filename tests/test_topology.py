import pytng


def test_n_molecules(GMX_REF_FILEPATH, GMX_REF_DATA):
    with pytng.TNGFile(GMX_REF_FILEPATH) as tng:
        assert tng.n_molecules == GMX_REF_DATA.n_molecules
