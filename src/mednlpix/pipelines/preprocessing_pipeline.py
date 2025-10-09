# src/mednlpix/pipelines/preprocessing_pipeline.py
import joblib
import datetime
from pathlib import Path
from pandas import DataFrame
from mednlpix.logger.logger import setup_logger
from mednlpix.utils.path_utils import find_project_root
from mednlpix.features.vectorization import TFIDFVectorizerWrapper, Word2VecVectorizer
from mednlpix.features.preprocessing.spacy_preprocessor import MedicalTextPreprocessor
from mednlpix.pipelines.registry_manager import update_registry


logger = setup_logger("preprocessing_pipeline")


class PreprocessingPipeline:
    """
    Production-ready preprocessing pipeline.
    Combines a text preprocessor and a vectorizer (TF-IDF or Word2Vec).
    """

    def __init__(self, method: str = "tfidf", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.preprocessor = MedicalTextPreprocessor(model_name="en_core_web_sm")

        if method == "tfidf":
            self.vectorizer = TFIDFVectorizerWrapper(**kwargs)
            logger.info(f"Initialized TF-IDF pipeline with parameters: {kwargs}")
        elif method == "word2vec":
            self.vectorizer = Word2VecVectorizer(**kwargs)
            logger.info(f"Initialized Word2Vec pipeline with parameters: {kwargs}")
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")

    def fit_transform(self, data: DataFrame):
        logger.info(f"Starting preprocessing training ({self.method}) on {len(data)} samples.")
        clean_texts = self.preprocessor.apply(data, text_column="medical_abstract")
        logger.info("Text cleaning and lemmatization completed. Starting vectorization.")
        features = self.vectorizer.fit_transform(clean_texts)
        logger.info(f"Vectorization completed ({self.method}) — feature matrix shape: {features.shape}.")
        return features

    def transform(self, data: DataFrame):
        logger.info(f"Applying preprocessing pipeline ({self.method}) to new dataset ({len(data)} rows).")
        clean_texts = self.preprocessor.apply(data, text_column="medical_abstract")
        logger.info("Text cleaning and lemmatization completed. Transforming to feature space.")
        features = self.vectorizer.transform(clean_texts)
        logger.info(f"Transformation completed — final matrix shape: {features.shape}.")
        return features

    def save(self, out_dir="src/mednlpix/models", bool_update_registry: bool = True):
        logger.info("Starting preprocessing pipeline saving process...")

        # Ensure paths are joined safely and platform-independently
        project_root = Path(find_project_root())
        full_path = project_root / out_dir
        full_path.mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"pipeline_{self.method}_{timestamp}.joblib"
        path = full_path / file_name

        # Serialize the pipeline
        joblib.dump(self, path)
        logger.info(f"Preprocessing pipeline successfully saved at: {path.resolve()}")

        if bool_update_registry:
            update_registry(model_path=str(path),model_type=f"PreprocessingPipeline_{self.method}")
            logger.info("Model registry updated.")

    def get_method(self):
        return self.method.upper()
