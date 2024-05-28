import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def make_h5ad(
    shape: tuple[int, int] | None = None,
    X: csr_matrix | np.ndarray | None = None,
    obs: pd.DataFrame | None = None,
    var: pd.DataFrame | None = None,
):
    if shape is not None:
        x = np.random.rand(*shape)

        mask = np.random.rand(*shape) < 0.7
        x[mask] = 0.0

        X = csr_matrix(x)

    if isinstance(X, np.ndarray):
        X = csr_matrix(X)
    X = X.astype(np.float32)

    n_obs = X.shape[0]
    n_var = X.shape[1]

    if obs is None:
        cell_types = list("ABCD")
        obs = pd.DataFrame(
            {
                "cell_type": [
                    cell_types[i] for i in np.random.randint(0, len(cell_types), n_obs)
                ],
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )
    else:
        assert obs.shape[0] == n_obs, "obs must have the same number of rows as X"

    if var is None:
        gene_names = [f"gene_{i}" for i in range(n_var)]
        var = pd.DataFrame(
            {
                "gene_name": gene_names,
            },
            index=gene_names,
        )
    else:
        assert var.shape[0] == n_var, "var must have the same number of rows as X"

    adata = ad.AnnData(X, obs, var)
    return adata
