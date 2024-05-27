import json
from functools import partial
from pathlib import Path
from typing import Callable, Iterable
from warnings import warn

import anndata as ad
import datasets
import numpy as np
from scipy.sparse import csr_matrix

from .base_builder import BaseBuilder, BuilderConfig

AdditionalFeatureCallbackType = Callable[[ad.AnnData, int], dict]


def _kirua_generator(
    idx_h5ad_files: list[tuple[int, Path]],
    backed_mode: str | None,
    chunk_size: int,
    use_cell_id_features: bool,
    use_raw: bool,
    gene_names: Iterable[str],
    gene_col_in_var_hint: str | list[str] | None,
    callback: AdditionalFeatureCallbackType | None,
    input_ids_dtype: np.dtypes.UIntDType,
):
    for dataset_idx, h5ad_file in idx_h5ad_files:
        adata = ad.read_h5ad(h5ad_file, backed=backed_mode)

        if isinstance(gene_col_in_var_hint, str) or isinstance(
            gene_col_in_var_hint, list
        ):
            if isinstance(gene_col_in_var_hint, str):
                gene_col_in_var_hint = [gene_col_in_var_hint]
            adata_gene_col = None
            for gene_col in gene_col_in_var_hint:
                if gene_col in adata.var.columns:
                    adata_gene_col = adata.obs[gene_col]
                    break
            if adata_gene_col is None:
                warn(
                    f"None of the hint columns {gene_col_in_var_hint} found in the file {h5ad_file}'s adata.var, "
                    f"fallback to adata.var_names instead"
                )
                adata_gene_col = adata.var_names
        elif gene_col_in_var_hint is None:
            adata_gene_col = adata.var_names

        gene2id = {gene: i for i, gene in enumerate(gene_names)}
        var_filter = adata_gene_col.isin(gene2id)

        filtered_gene_col = adata_gene_col[var_filter]
        input_ids_array = filtered_gene_col.map(gene2id).values.astype(input_ids_dtype)

        chunks = list(range(0, len(adata), chunk_size))

        X = adata.X if not use_raw else adata.raw.X

        for chunk in chunks:
            x_sparse: csr_matrix = X[chunk : chunk + chunk_size, var_filter]
            indptr = x_sparse.indptr
            data = x_sparse.data.astype(np.float32)
            indices = x_sparse.indices

            for i, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
                exprs = data[start:end]

                # even if the x is sparse, some zeros are still there, need to filter them
                nz = exprs > 0.0

                exprs = exprs[nz]
                col = indices[start:end][nz]
                input_ids = input_ids_array[col]

                # datasets.Array2D somehow very fast compare to datasets.Sequence(datasets.Value)
                features = {
                    "input_ids": input_ids,
                    "exprs": exprs,
                }

                cell_idx = int(i + chunk)
                if use_cell_id_features:
                    features["cell_idx"] = cell_idx
                    features["dataset_idx"] = dataset_idx

                if callback is not None:
                    features.update(callback(adata, cell_idx))
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
        gene_col_in_var_hint: str | list[str] | None = None,
        additional_features_callback: AdditionalFeatureCallbackType | None = None,
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

        self._gene_col_in_var_hint = gene_col_in_var_hint

        # whether to add cell id (dataset id + cell indices) as features
        self._use_cell_id_features = use_cell_id_features

        # callback to add additional features
        self._additional_features_callback = additional_features_callback

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

    @property
    def files(self):
        return sorted(list(self._files))

    def files_stat(self):
        files = self.files
        print(f"Number of files: {len(files)}")
        print("Files cell num:")
        cell_count = 0
        for f in files:
            adata = ad.read_h5ad(f, backed="r")
            print(f"{f}: {adata.n_obs}")
            cell_count += adata.n_obs
        print(f"Total cell count: {cell_count}")

    def build(self) -> datasets.Dataset:
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
            gene_col_in_var_hint=self._gene_col_in_var_hint,
            callback=self._additional_features_callback,
            input_ids_dtype=self._input_ids_dtype,
        )

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
