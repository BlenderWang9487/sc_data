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
        config = BuilderConfig()
        builder = KiruaBuilder(config, ["gene_0", "gene_1"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            files = []
            n_cells = 0
            for i in range(5):
                obs_size = np.random.randint(10, 20)
                var_size = np.random.randint(3, 10)

                n_cells += obs_size
                adata = make_h5ad(shape=(obs_size, var_size))
                file_name = tmpdir_path / f"adata_{i}.h5ad"
                adata.write_h5ad(file_name)
                files.append(file_name)

            ds = builder.add_files(files).build()

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


if __name__ == "__main__":
    unittest.main()
