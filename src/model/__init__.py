from typing import Dict

from .language_model import LanguageModel
from .bigram_model import BigramLanguageModel


MODELS: Dict[str, LanguageModel] = {
    'bigram': BigramLanguageModel,
}
