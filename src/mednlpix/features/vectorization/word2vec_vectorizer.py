# src/mednlpix/features/vectorization/word2vec_vectorizer.py

from gensim.models import Word2Vec
import numpy as np
from pandas import DataFrame
from mednlpix.features.vectorization.base_vectorizer import BaseVectorizer
from mednlpix.logger.logger import setup_logger

logger = setup_logger(__name__)


class Word2VecVectorizer(BaseVectorizer):
    """
    Word2Vec embedding generator for medical abstracts.
    Uses kwargs for flexible configuration (size, window, sg, epochs, etc.).
    """

    def __init__(self, **kwargs):
        """
        Initialize the Word2Vec vectorizer with default or custom parameters.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters passed to gensim.models.Word2Vec.
            Example: vector_size=200, window=10, min_count=3, sg=1, epochs=10
        """
        default_params = {
            "vector_size": 100,
            "window": 5,
            "min_count": 2,
            "workers": 4,
            "sg": 1,  # skip-gram (better for semantic relationships)
            "epochs": 10,
        }
        default_params.update(kwargs)
        self.params = default_params
        self.vector_size = default_params["vector_size"]
        self.model = None

        logger.info(f"Word2VecVectorizer initialized with parameters: {self.params}")

    # ------------------------------------------------------
    # Train the Word2Vec model
    # ------------------------------------------------------
    def fit(self, data: DataFrame, text_column: str = "cleaned_medical_abstract"):
        """
        Train a Word2Vec model on the corpus.

        Parameters
        ----------
        data : DataFrame
            Dataset containing the text column.
        text_column : str
            Column name containing cleaned and tokenized text.
        """
        logger.info(f"Training Word2Vec model on {len(data)} documents...")
        sentences = [text.split() for text in data[text_column]]
        self.model = Word2Vec(sentences=sentences, **self.params)

        logger.info(f"Word2Vec model trained successfully. Vocabulary size: {len(self.model.wv)} words.")
        return self

    # ------------------------------------------------------
    # Transform text into dense embeddings
    # ------------------------------------------------------
    def transform(self, data: DataFrame, text_column: str = "cleaned_medical_abstract"):
        """
        Convert each document into an averaged Word2Vec vector.

        Parameters
        ----------
        data : DataFrame
            Input dataset to transform.
        text_column : str
            Column name containing preprocessed text.

        Returns
        -------
        np.ndarray
            2D array of averaged Word2Vec embeddings.
        """
        if self.model is None:
            raise ValueError("Word2Vec model not trained. Call fit() first.")

        logger.info(f"Generating averaged embeddings for {len(data)} documents...")
        embeddings = []
        for text in data[text_column]:
            words = [w for w in text.split() if w in self.model.wv]
            if words:
                embeddings.append(np.mean(self.model.wv[words], axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))

        embeddings = np.array(embeddings)
        logger.info(f"Word2Vec transformation complete. Output shape: {embeddings.shape}")
        return embeddings

    # ------------------------------------------------------
    # Fit + Transform convenience method
    # ------------------------------------------------------
    def fit_transform(self, data: DataFrame, text_column: str = "cleaned_medical_abstract"):
        """
        Fit and transform in a single step.
        """
        return self.fit(data, text_column).transform(data, text_column)

    # ------------------------------------------------------
    # Optional: Save / Load model for reuse
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Optional: Save / Load model for reuse
    # ------------------------------------------------------
    def save(self, path):
        """
        Save the trained Word2Vec model to disk for later reuse.
        """
        if self.model is None:
            logger.warning("Word2Vec model not trained yet. Nothing to save.")
            return

        # Convert Path object to string for gensim compatibility
        self.model.save(str(path))
        logger.info(f"Word2Vec model saved successfully to {path}")

    def load(self, path):
        """
        Load a pre-trained Word2Vec model from disk.
        """
        from gensim.models import Word2Vec

        self.model = Word2Vec.load(str(path))
        self.vector_size = self.model.vector_size
        logger.info(f"Word2Vec model loaded successfully from {path}")
        return self
