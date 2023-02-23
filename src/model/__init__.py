from typing import Dict

from .language_model import LanguageModel
from .bigram_model import BigramLanguageModel
from .transformer_base import TransformerBase
from .basic_transformer import BasicTransformer



TRANSFORMERS: Dict[str, TransformerBase] = {
    'basic_transformer': BasicTransformer,
}
