# src/mednlpix/features/preprocessing/base.py
from abc import ABC, abstractmethod
from pandas import DataFrame


class BasePreprocessor(ABC):
    """
    Abstract base class for all data preprocessors.
    """

    @abstractmethod
    def fit(self, data: DataFrame):
        """Learn preprocessing parameters from training data."""
        pass

    @abstractmethod
    def transform(self, data: DataFrame) -> DataFrame:
        """Apply preprocessing transformations."""
        pass

    def fit_transform(self, data: DataFrame) -> DataFrame:
        """Fit on the data, then transform it."""
        self.fit(data)
        return self.transform(data)