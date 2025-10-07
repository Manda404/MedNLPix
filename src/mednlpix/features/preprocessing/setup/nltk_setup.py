# src/mednlpix/features/preprocessing/setup/nltk_setup.py

import shutil
from pathlib import Path
import nltk
from mednlpix.logger.logger import setup_logger

logger = setup_logger(__name__)

# =====================================================
# Configure NLTK resource directory (project-specific)
# =====================================================
NLTK_DATA_DIR = Path(__file__).resolve().parents[2] / "resources" / "nltk_data"

# Clean existing directory if partially downloaded
if NLTK_DATA_DIR.exists():
    logger.warning(f"Existing NLTK data directory found at {NLTK_DATA_DIR}. Deleting it...")
    shutil.rmtree(NLTK_DATA_DIR)

# Recreate directory
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Created fresh NLTK data directory at {NLTK_DATA_DIR}")

# Register this path so nltk looks here first
nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# =====================================================
# Download all required NLTK resources
# =====================================================
resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']

for resource in resources:
    try:
        logger.info(f"Downloading NLTK resource '{resource}' to {NLTK_DATA_DIR}...")
        nltk.download(resource, download_dir=str(NLTK_DATA_DIR))
    except Exception as e:
        logger.error(f"Failed to download resource '{resource}': {e}")

logger.info("All NLTK resources downloaded successfully.")
