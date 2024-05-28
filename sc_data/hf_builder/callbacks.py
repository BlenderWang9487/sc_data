import abc
from typing import Any

from anndata import AnnData


class BaseKiruaAdditionalFeaturesCallback(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, adata: AnnData, dataset_idx: int, cell_idx: int
    ) -> dict[str, Any]:
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
        self, adata: AnnData, dataset_idx: int, cell_idx: int
    ) -> dict[str, Any]:
        cell_type = adata.obs[self._cell_type_col].iloc[cell_idx]
        if not isinstance(cell_type, str):
            cell_type = self._na_value
        return {
            self._dataset_col: cell_type,
        }
