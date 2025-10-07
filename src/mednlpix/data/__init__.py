# src/mednlpix/__init__.py

from mednlpix.data.loader import get_medical_dataset
from mednlpix.data.splitter import split_medical_dataset

__all__ = ["get_medical_dataset","split_medical_dataset"]