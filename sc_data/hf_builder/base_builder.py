import abc
from dataclasses import dataclass
from typing import Generator

import datasets


@dataclass
class BuilderConfig:
    h5ad_backed: str | None = "r"
    use_raw: bool = False
    chunk_size: int = 5000
    num_workers: int = 1
    cache_dir: str | None = None

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")

        self.cache_dir = str(self.cache_dir) if self.cache_dir is not None else None


class BaseBuilder(abc.ABC):
    def __init__(self, config: BuilderConfig, **kwargs):
        self._config = config

    @abc.abstractmethod
    def build(self, **kwargs) -> datasets.Dataset | datasets.DatasetDict | Generator:
        raise NotImplementedError("build must be implemented")
