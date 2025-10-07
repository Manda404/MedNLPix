from pathlib import Path
from pandas import read_csv, DataFrame as PandasDataFrame
from mednlpix.logger.logger import setup_logger



def get_dataset_path(data_dir: str = "data/raw") -> Path:
    """
    Go one level up from the current working directory,
    then return the absolute path of the only CSV file inside `data_dir`.
    """
    logger = setup_logger(__name__)

    root_path = Path().resolve().parent
    data_path = root_path / data_dir
    logger.info(f"Searching for dataset in: {data_path}")

    csv_files = list(data_path.glob("*.csv"))

    if len(csv_files) == 0:
        logger.error(f"No CSV file found in {data_path}")
        raise FileNotFoundError(f"No CSV file found in {data_path}")
    if len(csv_files) > 1:
        logger.error(f"Multiple CSV files found in {data_path}")
        raise ValueError(f"Multiple CSV files found in {data_path}, expected only one.")

    dataset_path = csv_files[0]
    logger.info(f"Dataset found: {dataset_path.name}")
    return dataset_path


def load_dataset(dataset_path: Path) -> PandasDataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.
    """
    logger = setup_logger(__name__)

    logger.info(f"Loading dataset from: {dataset_path}")
    df = read_csv(dataset_path)
    logger.info(f"Dataset loaded successfully — the input data has {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def get_medical_dataset(data_dir: str = "data/raw", return_path: bool = False):
    """
    Main function that retrieves and loads the dataset.

    Parameters
    ----------
    data_dir : str, default="data/raw"
        Directory containing the dataset.
    return_path : bool, default=False
        If True, returns both the DataFrame and the dataset path.

    Returns
    -------
    PandasDataFrame
        The loaded dataset.
    OR
    tuple(PandasDataFrame, Path)
        If `return_path` is True, returns both the DataFrame and its path.
    """
    logger = setup_logger(__name__)

    try:
        dataset_path = get_dataset_path(data_dir)
        df = load_dataset(dataset_path)
        logger.info("Dataset successfully retrieved and loaded.")

        if return_path:
            return df, dataset_path
        return df

    except Exception as e:
        logger.exception(f"Error while loading dataset: {e}")
        raise