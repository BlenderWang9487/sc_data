import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sc_data.hf_builder import BuilderConfig, KiruaBuilder

from ._utils import make_h5ad


class TestKiruaBuilder(unittest.TestCase):
    def test_add_files(self):

        config = BuilderConfig()
        builder = KiruaBuilder(config, [])

        file_that_not_exist = Path("file_that_not_exist.h5ad")
        self.assertRaises(FileNotFoundError, builder.add_files, [file_that_not_exist])

        with tempfile.NamedTemporaryFile("r", suffix=".h5ad") as f:
            fname = Path(f.name)
            self.assertWarns(UserWarning, builder.add_files, [fname, fname])

    def test_build_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config = BuilderConfig(cache_dir=str(tmpdir_path / ".cache"))
            gene_names = ["gene_0", "gene_1"]
            builder = KiruaBuilder(config, gene_names=gene_names)
            files = []
            n_cells = 0
            ds_idx_to_test = 0
            cell_idx_to_test = 1

            for i in range(5):
                obs_size = np.random.randint(10, 20)
                var_size = np.random.randint(3, 10)

                n_cells += obs_size
                adata = make_h5ad(shape=(obs_size, var_size))
                if i == ds_idx_to_test and adata.X[cell_idx_to_test].nnz == 0:
                    # make sure there is at least one non-zero value in the cell to test
                    adata.X[cell_idx_to_test] = np.random.rand(var_size)
                file_name = tmpdir_path / f"adata_{i}.h5ad"
                adata.write_h5ad(file_name)
                files.append(file_name)

            ds = builder.add_files(files).build()
            ds.set_format("np")

            files = builder.files

            self.assertEqual(len(ds), n_cells, "Number of cells is not correct")

            self.assertIn(
                "cell_idx",
                ds.features.keys(),
                "by default cell_idx should be in features",
            )
            self.assertIn(
                "dataset_idx",
                ds.features.keys(),
                "by default dataset_idx should be in features",
            )

            # check the data in the dataset

            adata = ad.read_h5ad(files[ds_idx_to_test])
            adata = adata[:, gene_names]

            ds_0_cell_1_original_x = (
                (adata.X if not config.use_raw else adata.raw.X)[cell_idx_to_test]
                .toarray()
                .flatten()
            )
            non_zero = ds_0_cell_1_original_x[ds_0_cell_1_original_x > 0.0]

            record_in_ds = ds.filter(
                lambda x: x["dataset_idx"] == ds_idx_to_test
                and x["cell_idx"] == cell_idx_to_test
            )[0]

            self.assertTrue(
                (non_zero == record_in_ds["exprs"]).all(),
                f"Expression data is not correct, exprs in adata: {non_zero}, exprs in dataset: {record_in_ds['exprs']}",
            )


if __name__ == "__main__":
    unittest.main()
