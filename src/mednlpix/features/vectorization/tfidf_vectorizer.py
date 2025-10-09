# src/mednlpix/features/vectorization/tfidf_vectorizer.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from mednlpix.features.vectorization.base_vectorizer import BaseVectorizer
from mednlpix.logger.logger import setup_logger

logger = setup_logger(__name__)


class TFIDFVectorizerWrapper(BaseVectorizer):
    """
    TF-IDF Vectorizer for transforming medical abstracts into numerical features.
    Fully configurable through kwargs and reusable via save/load.
    """

    def __init__(self, **kwargs):
        """
        Initialize the TF-IDF vectorizer with default and custom parameters.

        Parameters
        ----------
        **kwargs : dict
            Optional parameters passed directly to sklearn.feature_extraction.text.TfidfVectorizer.
        """
        default_params = {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "sublinear_tf": True,
            "stop_words": "english",
        }
        default_params.update(kwargs)

        self.params = default_params
        self.vectorizer = TfidfVectorizer(**self.params)
        self.feature_names = None

        logger.info(f"TFIDFVectorizerWrapper initialized with params: {self.params}")

    # ------------------------------------------------------
    # Fit method
    # ------------------------------------------------------
    def fit(self, data: DataFrame, text_column: str = "cleaned_medical_abstract"):
        """
        Learn the TF-IDF vocabulary from the corpus.
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(data)} documents...")
        self.vectorizer.fit(data[text_column])
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF vocabulary learned. Total features: {len(self.feature_names)}")
        return self

    # ------------------------------------------------------
    # Transform method
    # ------------------------------------------------------
    def transform(self, data: DataFrame, text_column: str = "cleaned_medical_abstract"):
        """
        Transform documents into TF-IDF feature vectors.
        """
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise ValueError("TF-IDF vectorizer has not been fitted. Call fit() first.")

        logger.info(f"Transforming {len(data)} documents using TF-IDF...")
        features = self.vectorizer.transform(data[text_column])
        logger.info(f"TF-IDF transformation complete. Output shape: {features.shape}")
        return features

    # ------------------------------------------------------
    # Fit + Transform convenience method
    # ------------------------------------------------------
    def fit_transform(self, data: DataFrame, text_column: str = "cleaned_medical_abstract"):
        """
        Learn the TF-IDF vocabulary and transform the corpus in one step.
        """
        logger.info(f"Fitting and transforming {len(data)} documents with TF-IDF...")
        features = self.vectorizer.fit_transform(data[text_column])
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF fit_transform complete. Output shape: {features.shape}")
        return features

    # ------------------------------------------------------
    # Optional: Access feature names
    # ------------------------------------------------------
    def get_feature_names(self):
        """
        Retrieve the learned feature names (vocabulary terms).
        """
        if self.feature_names is None:
            raise ValueError("Vectorizer not fitted yet. No feature names available.")
        return self.feature_names

    # ------------------------------------------------------
    # Optional: Save / Load model for reuse
    # ------------------------------------------------------
    def save(self, path: str):
        """
        Save the trained TF-IDF vectorizer to disk for later reuse.
        """
        if not hasattr(self.vectorizer, "vocabulary_"):
            logger.warning("Vectorizer has not been fitted yet. Nothing to save.")
            return

        joblib.dump(self.vectorizer, path)
        logger.info(f"TF-IDF vectorizer saved successfully to {path}")

    def load(self, path: str):
        """
        Load a pre-trained TF-IDF vectorizer from disk.
        """
        self.vectorizer = joblib.load(path)
        self.feature_names = self.vectorizer.get_feature_names_out()
        logger.info(f"TF-IDF vectorizer loaded successfully from {path}")
        return self
