import abc
from typing import Any, Callable, Generator, Iterable

import pandas as pd
from anndata import AnnData

FeaturesDict = dict[str, Any]

#### Additional features callbacks section ####
AdditionalFeaturesCallbackType = Callable[
    [AnnData, int, int, FeaturesDict], FeaturesDict | None
]


class BaseKiruaAdditionalFeaturesCallback(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        adata: AnnData,
        dataset_idx: int,
        cell_idx: int,
        current_features: FeaturesDict,
    ) -> FeaturesDict | None:
        """
        Args:
            adata: AnnData object
            dataset_idx: index of the dataset
            cell_idx: index of the cell in the dataset
            current_features: current features dictionary of the cell

        Returns:
            FeaturesDict | None: updated features dictionary of the cell
                (which contains the new features and the old ones). If None is returned,
                means that the cell should be filtered out.
        """
        raise NotImplementedError


class KiruaCellTypeFeatureCallback(BaseKiruaAdditionalFeaturesCallback):
    def __init__(
        self,
        cell_type_col: str = "cell_type",
        dataset_col: str = "cell_type",
        na_value: str = "",
    ) -> None:
        self._cell_type_col = cell_type_col
        self._dataset_col = dataset_col
        self._na_value = na_value

    def __call__(
        self,
        adata: AnnData,
        dataset_idx: int,
        cell_idx: int,
        current_features: FeaturesDict,
    ) -> FeaturesDict | None:
        cell_type = adata.obs[self._cell_type_col].iloc[cell_idx]
        if not isinstance(cell_type, str):
            cell_type = self._na_value
        current_features[self._dataset_col] = cell_type
        return current_features


class KiruaSequentialCallback(BaseKiruaAdditionalFeaturesCallback):
    def __init__(self, callbacks: list[AdditionalFeaturesCallbackType]) -> None:
        self._callbacks = callbacks

    def __call__(
        self,
        adata: AnnData,
        dataset_idx: int,
        cell_idx: int,
        current_features: FeaturesDict,
    ) -> FeaturesDict | None:
        for callback in self._callbacks:
            current_features = callback(adata, dataset_idx, cell_idx, current_features)
            if current_features is None:
                return None
        return current_features


#### Gene names callbacks section ####
GeneNamesCallbackType = Callable[[AnnData, int], Iterable[str]]


class BaseKiruaGeneNamesCallback(abc.ABC):
    @abc.abstractmethod
    def __call__(self, adata: AnnData, dataset_idx: int) -> Iterable[str]:
        raise NotImplementedError


class KiruaGeneralGeneNamesCallback(BaseKiruaGeneNamesCallback):
    def __init__(self, gene_col_in_var_hint: str | list[str] | None = None) -> None:
        if isinstance(gene_col_in_var_hint, str):
            gene_col_in_var_hint = [gene_col_in_var_hint]
        elif isinstance(gene_col_in_var_hint, list) or gene_col_in_var_hint is None:
            pass
        else:
            raise ValueError(
                "gene_col_in_var_hint must be a str, a list of str, or None"
            )
        self._gene_col_in_var_hint = gene_col_in_var_hint

    def __call__(self, adata: AnnData, dataset_idx: int) -> Iterable[str]:
        from warnings import warn

        if self._gene_col_in_var_hint is None:
            return adata.var_names

        adata_gene_col = None
        for gene_col in self._gene_col_in_var_hint:
            if gene_col in adata.var.columns:
                adata_gene_col = adata.var[gene_col]
                break
        if adata_gene_col is None:
            warn(
                f"None of the hint columns {self._gene_col_in_var_hint} found in `adata.var`"
                f"fallback to adata.var_names instead"
            )
            adata_gene_col = adata.var_names
        return adata_gene_col
