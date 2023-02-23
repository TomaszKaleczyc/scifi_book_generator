from typing import Dict

from .language_model import LanguageModel
from .bigram_model import BigramLanguageModel
from .transformer import Transformer


MODELS: Dict[str, LanguageModel] = {
    'bigram': BigramLanguageModel,
    'transformer': Transformer,
}
