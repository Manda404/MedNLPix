# src/mednlpix/features/preprocessing/text_cleaner.py

from mednlpix.features.preprocessing.setup import nltk_setup ## This ensures NLTK resources are ready
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from mednlpix.logger.logger import setup_logger

logger = setup_logger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and preprocess a text string.
    """
    if not isinstance(text, str):
        return ""

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Remove HTML tags and URLs
    text = re.sub(r"<.*?>|http\S+", "", text)

    # Remove special characters, punctuation, and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Lowercase
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) >= 2]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)
