import abc
from typing import Optional, List

import nltk
from transformers import BertTokenizer


class IndexConstraintBase(abc.ABC):
    def __init__(self, tokenizer: Optional[BertTokenizer]):
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def is_allowed(self, token_id: int) -> bool:
        ...


class ComposedIndexConstraint(IndexConstraintBase):
    def __init__(self, constraints: List[IndexConstraintBase]):
        super().__init__(tokenizer=None)
        self.constraints = constraints

    def is_allowed(self, token_id: int) -> bool:
        allowed = True
        for c in self.constraints:
            if not c.is_allowed(token_id):
                allowed = False
                break
        return allowed


class SpecialIndexConstraint(IndexConstraintBase):
    def __init__(self, tokenizer: Optional[BertTokenizer]):
        super().__init__(tokenizer)
        self.special_token_ids = set([v for k, v in tokenizer.vocab.items() if '[' in k])
        self.special_token_ids = [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]

    def is_allowed(self, token_id: int) -> bool:
        return token_id not in self.special_token_ids


class StopwordIndexConstraint(IndexConstraintBase):
    def __init__(self, tokenizer: Optional[BertTokenizer]):
        super().__init__(tokenizer)
        stopwords = set(nltk.corpus.stopwords.words("english"))
        self.stopwords_ids = set([self.tokenizer.vocab[s] for s in stopwords if s in self.tokenizer.vocab])

    def is_allowed(self, token_id: int) -> bool:
        return token_id not in self.stopwords_ids
