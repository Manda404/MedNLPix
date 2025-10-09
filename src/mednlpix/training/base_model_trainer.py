# src/mednlpix/training/base_model_trainer.py

import joblib
import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from mednlpix.logger.logger import setup_logger
from mednlpix.inference.preprocessing_pipeline import PreprocessingPipeline



class BaseModelTrainer(ABC):
    """
    Base class for all boosting model trainers (CatBoost, XGBoost, LightGBM).

    This abstract class defines the **common interface and shared logic** 
    for all boosting models. It handles data preprocessing, model training 
    orchestration, metrics extraction, and model saving — while delegating 
    framework-specific logic (train, save, visualize) to subclasses.

    ---
    ### Key Responsibilities
    - **Preprocessing**: Fit and transform text data using the provided preprocessing pipeline.
    - **Model Training (abstract)**: Define training logic in subclasses.
    - **Metrics Extraction**: Retrieve learning metrics for plotting or monitoring.
    - **Model Persistence (abstract)**: Save trained model artifacts.
    - **Visualization (abstract)**: Plot learning curves for model comparison.

    ---
    ### Parameters
    - `model` : Any  
        Initialized model instance (CatBoostClassifier, XGBClassifier, LGBMClassifier, etc.)

    - `train_df` : pd.DataFrame  
        Training dataset.

    - `valid_df` : pd.DataFrame  
        Validation dataset.

    - `preprocessor` : PreprocessingPipeline  
        Preprocessing pipeline instance handling text/vectorization.

    - `**kwargs` : dict  
        Optional configuration parameters:
        - `text_col` (str): Name of text column (default: `"medical_abstract"`)
        - `target_col` (str): Name of target column (default: `"condition_label"`)
    """

    def __init__(self, model, train_df, valid_df, preprocessor: PreprocessingPipeline, **kwargs):
        self.model = model
        self.train_df = train_df
        self.valid_df = valid_df
        self.preprocessor = preprocessor
        self.text_col = kwargs.get("text_col", "medical_abstract")
        self.target_col = kwargs.get("target_col", "condition_label")
        self.framework = model.__class__.__name__.lower()
        self.logger = setup_logger(f"{self.framework}_trainer")

    # ==========================================================
    # 1. Preprocessing
    # ==========================================================
    def preprocess(self, **kwargs):
        """
        Fit and transform the preprocessing pipeline on the training dataset,
        then transform the validation dataset.

        Saves the preprocessing pipeline to disk and optionally updates the registry.

        Parameters
        ----------
        **kwargs : dict
            - `bool_update_registry` (bool): whether to update registry after saving (default: True)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed training and validation matrices.
        """
        self.logger.info(f"Fitting {self.preprocessor.get_method()} pipeline on training data...")

        X_train = self.preprocessor.fit_transform(self.train_df)
        X_valid = self.preprocessor.transform(self.valid_df)

        self.preprocessor.save(bool_update_registry=kwargs.get("bool_update_registry", True))
        self.logger.info(f"{self.preprocessor.get_method()} preprocessing pipeline saved successfully.")
        return X_train, X_valid

    # ==========================================================
    # 2. Model Training (abstract)
    # ==========================================================
    @abstractmethod
    def train(self, X_train, y_train, X_valid, y_valid):
        """
        Train the model — must be implemented by each subclass.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        X_valid : np.ndarray
            Validation features.
        y_valid : np.ndarray
            Validation labels.
        """
        pass

    # ==========================================================
    # 3. Get Training Metrics
    # ==========================================================
    def get_training_metrics(self, model: Any) -> Tuple[List[float], List[float], str]:
        """
        Extract training and validation metrics depending on the boosting framework.

        Parameters
        ----------
        model : Any
            Trained boosting model.

        Returns
        -------
        Tuple[List[float], List[float], str]
            (train_scores, valid_scores, metric_name)
        """
        framework: str = self.model.__class__.__name__.lower()

        if framework == "lgbmclassifier":
            self.logger.info("Extracting training metrics from LightGBM model...")
            evals_result = model.evals_result_
            train_key, valid_key = "train", "valid"

        elif framework == "xgbclassifier":
            self.logger.info("Extracting training metrics from XGBoost model...")
            evals_result = model.evals_result()
            train_key, valid_key = "validation_0", "validation_1"

        elif framework == "catboostclassifier":
            self.logger.info("Extracting training metrics from CatBoost model...")
            evals_result = model.get_evals_result()
            train_key, valid_key = "learn", "validation"

        else:
            raise ValueError("Framework must be 'lightgbm', 'xgboost', or 'catboost'.")

        if not evals_result:
            raise ValueError("Evaluation results are not available for this model.")

        metric_name = list(evals_result[train_key].keys())[0]
        self.logger.info(f"Tracking evaluation metric: {metric_name}")

        return (
            evals_result[train_key][metric_name],
            evals_result[valid_key][metric_name],
            metric_name,
        )

    # ==========================================================
    # 4. Get Model
    # ==========================================================
    def get_model(self) -> Any:
        """
        Return the internal trained model instance.

        Returns
        -------
        Any
            The trained model.
        """
        return self.model

    # ==========================================================
    # 5. Save model
    # ==========================================================
    @abstractmethod
    def save_model(self, out_dir: str = "/src/mednlpix/models"):
        """
        Save the trained model to disk.

        Parameters
        ----------
        out_dir : str, optional
            Output directory for model artifacts.
        """
        pass

    # ==========================================================
    # 6. Visualization placeholder
    # ==========================================================
    @abstractmethod
    def visualize_learning_curves(self):
        """
        Visualize learning curves (loss, accuracy, etc.)
        Each subclass must implement its own visualization logic.
        """
        pass
