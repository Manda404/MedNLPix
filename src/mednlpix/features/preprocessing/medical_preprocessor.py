# src/mednlpix/features/preprocessing/medical_preprocessor.py
from pandas import DataFrame
from mednlpix.logger.logger import setup_logger
from mednlpix.features.preprocessing.base import BasePreprocessor
from mednlpix.features.preprocessing.text_cleaner import clean_text

logger = setup_logger(__name__)


class MedicalTextPreprocessor(BasePreprocessor):
    """
    Preprocessor specialized for cleaning and enriching medical abstract data.

    Steps:
    - Clean text using `clean_text`
    - Add a text length column
    """

    def __init__(
        self,
        text_column: str = "medical_abstract",
        cleaned_column: str = "cleaned_medical_abstract",
        length_column: str = "clean_original_length",
    ):
        self.text_column = text_column
        self.cleaned_column = cleaned_column
        self.length_column = length_column

    def fit(self, data: DataFrame):
        """No fitting required for this preprocessor, but keep API consistent."""
        logger.info(f"Fitting preprocessor on column '{self.text_column}'...")
        if self.text_column not in data.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame.")
        logger.info("No fitting required for this text preprocessor.")
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        """Apply text cleaning and add a length column."""
        logger.info("Starting preprocessing of medical abstracts...")

        if self.text_column not in data.columns:
            raise ValueError(f"Column '{self.text_column}' not found in DataFrame.")

        data = data.copy()
        data[self.cleaned_column] = data[self.text_column].apply(clean_text)
        data[self.length_column] = data[self.cleaned_column].apply(
            lambda x: len(str(x).split())
        )

        logger.info(
            f"Preprocessing completed — dataset shape: {data.shape}, "
            f"mean text length: {data[self.length_column].mean():.2f} words."
        )

        return data
