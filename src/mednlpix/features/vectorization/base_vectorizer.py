# src/mednlpix/features/vectorization/base_vectorizer.py

from abc import ABC, abstractmethod
from pandas import DataFrame

class BaseVectorizer(ABC):
    """
    Abstract base class for all vectorizers (TF-IDF, Word2Vec, etc.).
    Defines a consistent interface for fit, transform, and fit_transform methods.
    """

    @abstractmethod
    def fit(self, data: DataFrame, text_column: str):
        """
        Fit the vectorizer on a dataset.
        """
        pass

    @abstractmethod
    def transform(self, data: DataFrame, text_column: str):
        """
        Transform a dataset into numerical features.
        """
        pass

    def fit_transform(self, data: DataFrame, text_column: str):
        """
        Default implementation of fit + transform.
        Subclasses can override if optimization is possible.
        """
        self.fit(data, text_column)
        return self.transform(data, text_column)