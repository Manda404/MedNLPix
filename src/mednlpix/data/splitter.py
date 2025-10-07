from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from mednlpix.logger.logger import setup_logger

def split_medical_dataset(
    df: DataFrame,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    random_state: int = 42,
    shuffle: bool = True,
    **kwargs,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split a pandas DataFrame into train/validation/test sets.

    Parameters
    ----------
    df : DataFrame
        Full dataset (all columns preserved).
    train_size : float, optional
        Proportion for training set. Defaults to 0.7 if not provided.
    valid_size : float, optional
        Proportion for validation set. Defaults to 0.15 if not provided.
    test_size : float, optional
        Proportion for test set. Defaults to 0.15 if not provided.
    random_state : int, default=42
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    **kwargs :
        Extra arguments passed to sklearn's `train_test_split` 
        (e.g., stratify=df["target"]).

    Returns
    -------
    train_df, valid_df, test_df : Tuple[DataFrame, DataFrame, DataFrame]
        Three pandas DataFrames with all original columns.
    """
    logger = setup_logger(__name__)

    # Default proportions if not provided
    if train_size is None and valid_size is None and test_size is None:
        train_size, valid_size, test_size = 0.7, 0.15, 0.15

    # Normalize if proportions don't sum to 1
    total = sum(x for x in [train_size, valid_size, test_size] if x is not None)
    if abs(total - 1.0) > 1e-6:  # allow float precision tolerance
        logger.warning(f"Proportions do not sum to 1 (total={total:.2f}). Normalizing...")
        train_size = (train_size or 0.7) / total
        valid_size = (valid_size or 0.15) / total
        test_size = (test_size or 0.15) / total

    # First split: train vs temp (valid+test)
    train_df, temp_df = train_test_split(
        df, train_size=train_size, random_state=random_state, shuffle=shuffle, **kwargs
    )

    # Then split temp into valid/test
    relative_valid_size = valid_size / (valid_size + test_size)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=relative_valid_size,
        random_state=random_state,
        shuffle=shuffle,
        **kwargs,
    )

    logger.info(
        f"Split - Train: {train_df.shape}, Valid: {valid_df.shape}, Test: {test_df.shape}"
    )
    return train_df, valid_df, test_df