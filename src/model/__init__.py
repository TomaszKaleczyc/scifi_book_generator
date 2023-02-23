from typing import Dict

from .transformer_base import TransformerBase
from .language_model import LanguageModel
from .bigram_model import BigramLanguageModel
from .single_head_transformer import SingleHeadTransformer
from .multi_head_transformer import MultiHeadTransformer


TRANSFORMERS: Dict[str, TransformerBase] = {
    'single_head_transformer': SingleHeadTransformer,
    'multi_head_transformer': MultiHeadTransformer
}
