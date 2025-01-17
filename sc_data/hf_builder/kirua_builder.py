import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Iterable
from warnings import warn

import anndata as ad
import datasets
import numpy as np
import pandas as pd

# from 0.11.0, the CSRDataset is moved to anndata.abc and no longer in anndata._core.sparse_dataset
from packaging.version import Version

if Version(ad.__version__) < Version("0.11.0"):
    from anndata._core.sparse_dataset import CSRDataset
else:
    from anndata.abc import CSRDataset

from scipy.sparse import csr_matrix

from .base_builder import BaseBuilder, BuilderConfig
from .callbacks import (
    AdditionalFeaturesCallbackType,
    GeneNamesCallbackType,
    KiruaGeneralGeneNamesCallback,
)


def _mem_efficient_csr_matrix_filtering(
    X: csr_matrix,
    row_start: int,
    row_end: int,
    col_bool_mask: np.ndarray,
):
    # **Note: X must be a csr_matrix to make the following code memory efficient**

    # row filtering on backed HDF5 CSR won't break the backed mode, and will transfer the selected data to memory
    x_sparse = X[row_start:row_end]

    # if X is not csr_matrix, this will make sure x_sparse is csr_matrix
    x_sparse = csr_matrix(x_sparse[:, col_bool_mask])
    return x_sparse


def _kirua_generator(
    idx_h5ad_files: list[tuple[int, Path]],
    backed_mode: str | None,
    chunk_size: int,
    use_cell_id_features: bool,
    use_raw: bool,
    gene_names: Iterable[str],
    additional_features_callback: AdditionalFeaturesCallbackType | None,
    gene_names_callback: GeneNamesCallbackType,
    input_ids_dtype: np.dtypes.UIntDType,
):
    for dataset_idx, h5ad_file in idx_h5ad_files:
        # first check the file is empty or not
        if os.stat(h5ad_file).st_size == 0:
            warn(
                f"dataset idx: {dataset_idx}, File {h5ad_file} is empty, skip this file"
            )
            continue

        adata = ad.read_h5ad(h5ad_file, backed=backed_mode)

        # get the gene names from the adata
        adata_gene_col = pd.Series(
            gene_names_callback(adata if not use_raw else adata.raw, dataset_idx)
        )

        # filter the gene names based on `gene_names`
        gene2id = {gene: i for i, gene in enumerate(gene_names)}
        var_filter = adata_gene_col.isin(gene2id)
        print(
            f"dataset idx: {dataset_idx}, {var_filter.sum()}/{len(var_filter)} genes are found, ratio: {var_filter.mean():.2%}"
        )

        filtered_gene_col = adata_gene_col[var_filter]
        input_ids_array = filtered_gene_col.map(gene2id).values.astype(input_ids_dtype)

        chunks = list(range(0, len(adata), chunk_size))

        X = adata.X if not use_raw else adata.raw.X
        assert X.shape[1] == len(
            adata_gene_col
        ), f"X and gene names size mismatch, {X.shape[1]} != {len(adata_gene_col)}"
        if backed_mode is not None and not isinstance(X, CSRDataset):
            warn(
                f"You are using backed mode: {backed_mode}, but the X is not in CSR format"
                f" (it's {type(X)}), this will make the filtering process slower"
                " since the row seletion will break the backed mode and transfer all the data to memory."
            )

        for chunk in chunks:
            x_sparse = _mem_efficient_csr_matrix_filtering(
                X=X,
                row_start=chunk,
                row_end=(chunk + chunk_size),
                col_bool_mask=var_filter,
            )
            indptr = x_sparse.indptr
            data = x_sparse.data.astype(np.float32)
            indices = x_sparse.indices

            se_list = list(zip(indptr[:-1], indptr[1:]))
            for i, (start, end) in enumerate(se_list):
                exprs = data[start:end]

                # even if the x is sparse, some zeros are still there, need to filter them
                nz = exprs > 0.0

                exprs = exprs[nz]
                col = indices[start:end][nz]
                input_ids = input_ids_array[col]

                # datasets.Array2D somehow very fast compare to datasets.Sequence(datasets.Value)
                #  if we "strickly specify" the datasets.Features. But if we don', this 1d np.ndarray
                #  will be auto detected as datasets.Sequence(datasets.Value) and keep the same speed as datasets.Array2D
                # This is a bit weird, but it's fine for us to not specify Features type for now
                features = {
                    "input_ids": input_ids,
                    "exprs": exprs,
                }

                cell_idx = int(i + chunk)
                if use_cell_id_features:
                    features["cell_idx"] = cell_idx
                    features["dataset_idx"] = dataset_idx

                if additional_features_callback is not None:
                    # current features are also passed to the callback, since some
                    # features may need input_ids or exprs to generate new features
                    # for example, num of expressed genes, or species (can infer from the input_ids)
                    features = additional_features_callback(
                        adata, dataset_idx, cell_idx, features
                    )

                    # this condition make features_callback more flexible
                    # now features_callback can also be a filter to skip some cells
                    if features is None:
                        continue
                yield features

            del x_sparse
        del adata


class KiruaBuilder(BaseBuilder):
    AVAILABLE_INPUT_IDS_DTYPES: list = [np.uint16, np.uint32]
    GENE_NAME_FILE: str = "gene_names.txt"
    FILES_FILE: str = "files.txt"
    BUILDER_CONFIG_FILE: str = "builder_config.json"

    def __init__(
        self,
        config: BuilderConfig,
        gene_names: Iterable[str],
        use_cell_id_features: bool = True,
        additional_features_callback: AdditionalFeaturesCallbackType | None = None,
        gene_names_callback: GeneNamesCallbackType | None = None,
    ):
        super().__init__(config)
        self._files = set()
        self._gene_names = list(gene_names)

        # infer input_ids dtype based on the num of genes (if less than 2^16, use uint16, else use uint32)
        self._input_ids_dtype = None
        for t in KiruaBuilder.AVAILABLE_INPUT_IDS_DTYPES:
            if len(gene_names) <= (np.iinfo(t).max) + 1:
                self._input_ids_dtype = t
                break
        if self._input_ids_dtype is None:
            raise ValueError("Too many genes, cannot fit in uint16 or uint32")

        # whether to add cell id (dataset id + cell indices) as features
        self._use_cell_id_features = use_cell_id_features

        # callback to add additional features
        self._additional_features_callback = additional_features_callback

        # callback to get gene names
        self._gene_names_callback = (
            gene_names_callback
            if gene_names_callback is not None
            else KiruaGeneralGeneNamesCallback()
        )

    @property
    def additional_features_callback(self):
        return self._additional_features_callback

    @additional_features_callback.setter
    def additional_features_callback(
        self, callback: AdditionalFeaturesCallbackType | None
    ):
        if self._additional_features_callback is not None:
            warn("Overwriting the old additional_features_callback")
        self._additional_features_callback = callback

    @property
    def gene_names_callback(self):
        return self._gene_names_callback

    @gene_names_callback.setter
    def gene_names_callback(self, callback: GeneNamesCallbackType | None):
        if self._gene_names_callback is not None:
            warn("Overwriting the old gene_names_callback")
        self._gene_names_callback = callback

    def add_files(self, files: Iterable[str]) -> "KiruaBuilder":
        for f in files:
            file = Path(f)
            if not file.exists():
                raise FileNotFoundError(f"File {f} does not exist")
            file = file.resolve()

            if file in self._files:
                warn(
                    f"File {f} is already in the set of files, make sure this is what you want"
                )
            self._files.add(file)
        return self

    def filter_files(self, filter_func: Callable[[str], bool]) -> "KiruaBuilder":
        self._files = set(filter(filter_func, self._files))
        return self

    @property
    def files(self):
        return sorted(list(self._files))

    def files_stat(self) -> dict[str, Any]:
        files = self.files
        num_files = len(files)
        cell_count = 0
        cell_per_files = dict()
        for f in files:
            adata = ad.read_h5ad(f, backed="r")
            cell_per_files[f] = adata.n_obs
            cell_count += adata.n_obs
        total_cells = cell_count
        return {
            "num_files": num_files,
            "total_cells": total_cells,
            "cell_per_files": cell_per_files,
        }

    def build(self, return_generator: bool = False) -> datasets.Dataset | Generator:
        """Build the dataset or return the generator

        Args:
            return_generator (bool, optional): return a generator for `datasets.Dataset.from_generator`
                *Note: you should pass gen_kwargs={"idx_h5ad_files": list[tuple[int, str]]} to `from_generator`
                if you set return_generator to True. Defaults to False.

        Returns:
            datasets.Dataset | Generator: the built dataset or the generator
        """
        files = self.files
        assert len(files) > 0, "No files added"
        print(f"Start building dataset from {len(files)} files")
        print("files:")
        if len(files) > 5:
            print("\n".join([str(f) for f in files[:4]] + ["..."] + [str(files[-1])]))
        else:
            print("\n".join([str(f) for f in files]))

        idx_files = [(idx, str(f)) for idx, f in enumerate(files)]

        generator = partial(
            _kirua_generator,
            backed_mode=self._config.h5ad_backed,
            chunk_size=self._config.chunk_size,
            use_cell_id_features=self._use_cell_id_features,
            use_raw=self._config.use_raw,
            gene_names=self._gene_names,
            additional_features_callback=self._additional_features_callback,
            gene_names_callback=self._gene_names_callback,
            input_ids_dtype=self._input_ids_dtype,
        )

        if return_generator:
            return generator

        ds = datasets.Dataset.from_generator(
            generator,
            cache_dir=self._config.cache_dir,
            gen_kwargs={"idx_h5ad_files": idx_files},
            num_proc=self._config.num_workers,
        )

        return ds

    def save_dataset_meta(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / KiruaBuilder.GENE_NAME_FILE, "w") as f:
            f.write("\n".join(self._gene_names))

        with open(p / KiruaBuilder.FILES_FILE, "w") as f:
            f.write("\n".join([str(file) for file in self._files]))

        with open(p / KiruaBuilder.BUILDER_CONFIG_FILE, "w") as f:
            json.dump(self._config.__dict__, f, indent=2)
