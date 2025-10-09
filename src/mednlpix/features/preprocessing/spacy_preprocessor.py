# src/mednlpix/features/preprocessing/spacy_preprocessor.py

import spacy
from pandas import DataFrame
from mednlpix.logger.logger import setup_logger

logger = setup_logger(__name__)

class MedicalTextPreprocessor:
    """
    Lightweight text preprocessing pipeline using spaCy.
    Cleans and lemmatizes medical abstracts efficiently.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Load the spaCy language model once.
        """
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            self.nlp = spacy.load(model_name, disable=["ner", "parser", "textcat"])
        except OSError:
            logger.warning(f"Model '{model_name}' not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name, disable=["ner", "parser", "textcat"])

        logger.info("spaCy model loaded successfully.")

    def preprocess_text(self, text: str) -> str:
        """
        Clean and lemmatize a single text string.

        Steps:
        - Lowercase
        - Tokenize
        - Remove stopwords, punctuation, and numbers
        - Lemmatize tokens
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        doc = self.nlp(text.lower())

        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and token.is_alpha and len(token) > 2
        ]
        return " ".join(tokens)

    def apply(self, data: DataFrame, text_column: str = "medical_abstract") -> DataFrame:
        """
        Apply preprocessing to an entire DataFrame column.
        Returns a new column: 'cleaned_medical_abstract'.
        """
        if text_column not in data.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        logger.info(f"Preprocessing column '{text_column}' using spaCy...")

        # Make an explicit copy to avoid SettingWithCopyWarning
        processed_df = data.copy()

        processed_df["cleaned_medical_abstract"] = processed_df[text_column].apply(self.preprocess_text)

        avg_len = processed_df["cleaned_medical_abstract"].apply(lambda x: len(x.split())).mean()
        logger.info(f"Preprocessing completed. Average token count: {avg_len:.2f}")

        return processed_df
