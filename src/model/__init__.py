from typing import Dict

from .language_model import LanguageModel
from .bigram_model import BigramLanguageModel
from .basic_transformer import BasicTransformer


TRANSFORMERS: Dict[str, LanguageModel] = {
    'basic_transformer': BasicTransformer,
}
